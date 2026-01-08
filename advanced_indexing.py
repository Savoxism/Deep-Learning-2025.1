import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from pymilvus import MilvusClient
from utils.models.embedder import EmbeddingModel
from advanced_chunking import LayoutAwareChunker 
from src.helper import setup_logger

"""
Examples:

# 1) Index from CSV (expects columns: cid, text)
python index.py --input "data/filtered_corpus.csv" --output "vector_db/VN_legal.db" --collection "default" \
  --model-name "intfloat/multilingual-e5-base" --batch-size 64

# 2) Index from pre-chunked JSON (LayoutAwareChunker output)
python index.py --input "data/final_outputs/Contract_1.json" --input-type json --output "vector_db/VN_legal.db" \
  --collection "default" --model-name "intfloat/multilingual-e5-base" --batch-size 64
"""

logger = setup_logger(log_file="indexing.log")


def embed_batch(texts: List[str], embedding_model: EmbeddingModel, batch_size: int = 32) -> List[List[float]]:
    """
    Create embeddings for a batch of texts, with robust dimension handling and padding fallbacks.
    """
    all_embeddings: List[List[float]] = []
    PROCESSING_BATCH = 512

    # Detect embedding dim once
    dummy_emb = embedding_model.encode(["test"], batch_size=1)
    if hasattr(dummy_emb, "shape"):
        EMBEDDING_DIM = int(dummy_emb.shape[1])
    else:
        EMBEDDING_DIM = len(dummy_emb[0])

    logger.info(f"Detected Model Dimension: {EMBEDDING_DIM}")

    for i in tqdm(range(0, len(texts), PROCESSING_BATCH), desc="Encoding Progress"):
        batch_texts = texts[i : i + PROCESSING_BATCH]

        try:
            batch_emb = embedding_model.encode(batch_texts, batch_size=batch_size)

            # Normalize to list-of-vectors
            if isinstance(batch_emb, np.ndarray):
                batch_emb_list = batch_emb.tolist()
            else:
                batch_emb_list = [e.tolist() if isinstance(e, np.ndarray) else list(e) for e in batch_emb]

            # Handle mismatch
            if len(batch_emb_list) != len(batch_texts):
                logger.warning(
                    f"⚠️ Batch mismatch at index {i}! Input: {len(batch_texts)}, Output: {len(batch_emb_list)}"
                )
                missing = len(batch_texts) - len(batch_emb_list)
                batch_emb_list.extend([[0.0] * EMBEDDING_DIM for _ in range(max(0, missing))])

            # Handle per-vector dim mismatch (rare, but defensive)
            fixed: List[List[float]] = []
            for v in batch_emb_list:
                if len(v) == EMBEDDING_DIM:
                    fixed.append(v)
                elif len(v) > EMBEDDING_DIM:
                    fixed.append(v[:EMBEDDING_DIM])
                else:
                    fixed.append(v + [0.0] * (EMBEDDING_DIM - len(v)))

            all_embeddings.extend(fixed)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"❌ Error encoding batch starting at {i}: {str(e)}")
            all_embeddings.extend([[0.0] * EMBEDDING_DIM for _ in range(len(batch_texts))])

    return all_embeddings


def _ensure_collection(milvus_client: MilvusClient, collection_name: str, dim: int) -> None:
    """
    Drop + recreate collection for a clean indexing run.
    """
    if milvus_client.has_collection(collection_name):
        logger.warning(f"Collection '{collection_name}' exists. Dropping...")
        milvus_client.drop_collection(collection_name)

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="COSINE",
        auto_id=True,
        enable_dynamic_field=True,  # store full metadata without rigid schema
    )


def indexing(
    records: List[Dict[str, Any]],
    embeddings: List[List[float]],
    milvus_client: MilvusClient,
    collection_name: str,
) -> None:
    logger.info("=" * 60)
    logger.info(f"SAVING TO MILVUS: {collection_name}")

    if not records:
        logger.error("No records to index!")
        return
    if not embeddings:
        logger.error("No embeddings to save!")
        return
    if len(records) != len(embeddings):
        logger.error("Mismatch count records vs embeddings! Indexing aborted to prevent data corruption.")
        return

    dim = len(embeddings[0])
    logger.info(f"Final Embedding dimension: {dim}")

    _ensure_collection(milvus_client, collection_name, dim)

    # Prepare data: vector field + dynamic fields for metadata
    data: List[Dict[str, Any]] = []
    logger.info(f"Preparing {len(records)} records...")

    for rec, vec in zip(records, embeddings):
        # Minimal dynamic payload: always include raw_text for debugging + retrieval display.
        # Store full record as dynamic fields (Milvus supports nested JSON-like maps via dynamic fields).
        item = {
            "vector": vec,
            "raw_text": rec.get("raw_text") or rec.get("text") or "",
            "chunk_id": rec.get("chunk_id") or str(rec.get("cid", "")),
        }
        # Attach all remaining metadata as dynamic fields
        # (Milvus dynamic fields accept extra keys; keep them flat-ish)
        for k, v in rec.items():
            if k in ("raw_text", "text"):
                continue
            item[k] = v

        data.append(item)

    logger.info(f"Inserting {len(data)} records...")
    insert_batch = 1000
    for i in tqdm(range(0, len(data), insert_batch), desc="Inserting to DB"):
        milvus_client.insert(collection_name=collection_name, data=data[i : i + insert_batch])

    logger.info("✅ Save complete!")


def _records_from_layoutaware_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load LayoutAwareChunker output (list of FinalChunkOutput dicts) from .json.
    """
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            # allow single object fallback
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("JSON must be a list of chunk objects.")
        return payload
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON chunks from {json_path}: {e}")


def _records_from_csv(
    csv_path: Path,
    chunker: Optional[LayoutAwareChunker],
    intermediate_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """
    CSV mode:
      - If --chunking layoutaware: interpret each row text as a document and chunk with LayoutAwareChunker
        (note: LayoutAwareChunker expects HTML tables if present; works for plain text too).
      - If --chunking none: store each row as one record (no chunking).
    Requires columns: text, cid (cid optional if not present).
    """
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    if "cid" not in df.columns:
        logger.warning("CSV does not contain 'cid' column; using row index as cid.")
        df["cid"] = df.index.astype(str)

    records: List[Dict[str, Any]] = []

    if chunker is None:
        # No chunking: 1 row = 1 record
        for _, row in df.iterrows():
            text_str = str(row["text"]).strip()
            if not text_str:
                continue
            records.append(
                {
                    "chunk_id": str(row["cid"]),
                    "source_doc": str(row["cid"]),
                    "chunk_type": "text",
                    "raw_text": text_str,
                    "cid": str(row["cid"]),
                }
            )
        return records

    # LayoutAwareChunker chunking per-row treated as a "document"
    # We write each row to a temporary .md file so the same pipeline applies (tables extraction etc.).
    tmp_root = (intermediate_dir or csv_path.parent / "tmp_docs")
    tmp_root.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking Docs (LayoutAware)"):
        cid = str(row["cid"])
        text_str = str(row["text"]).strip()
        if not text_str:
            continue

        tmp_md = tmp_root / f"{cid}.md"
        tmp_md.write_text(text_str, encoding="utf-8")

        try:
            doc_chunks = chunker.process_document(tmp_md, intermediate_dir=intermediate_dir)
            # attach cid for traceability
            for ch in doc_chunks:
                ch["cid"] = cid
            records.extend(doc_chunks)
        except Exception as e:
            logger.error(f"❌ Chunking failed for cid={cid}: {e}")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/filtered_corpus.csv", help="CSV path or JSON chunks path")
    parser.add_argument("--input-type", choices=["csv", "json"], default=None, help="Force input type")
    parser.add_argument("--output", default="vector_db/default.db", help="Milvus DB Path")
    parser.add_argument("--collection", default="deep_learning_proj", help="Collection Name")
    parser.add_argument("--model-name", default="AITeamVN/Vietnamese_Embedding")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding inference batch size")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Embedding model max sequence length")

    # Chunking controls
    parser.add_argument(
        "--chunking",
        choices=["layoutaware", "none"],
        default="layoutaware",
        help="Chunking strategy for CSV input. JSON input is assumed already chunked.",
    )
    parser.add_argument("--intermediate-dir", default="data/intermediate_outputs", help="Intermediate outputs dir")

    args = parser.parse_args()

    input_path = Path(args.input)
    if args.input_type is None:
        inferred = "json" if input_path.suffix.lower() == ".json" else "csv"
        args.input_type = inferred

    logger.info(f"START INDEXING: {args.input} (type={args.input_type})")
    logger.info(f"MODEL: {args.model_name}")

    # Create DB directory
    db_dir = os.path.dirname(args.output)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model = EmbeddingModel(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        device=DEVICE,
    )
    milvus_client = MilvusClient(uri=args.output)

    intermediate_dir = Path(args.intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Build records
    if args.input_type == "json":
        records = _records_from_layoutaware_json(input_path)
    else:
        chunker = LayoutAwareChunker() if args.chunking == "layoutaware" else None
        records = _records_from_csv(input_path, chunker, intermediate_dir)

    # Filter empties
    valid_records = []
    for r in records:
        txt = (r.get("raw_text") or r.get("text") or "").strip()
        if txt:
            valid_records.append(r)
    records = valid_records
    logger.info(f"Total valid records to embed: {len(records)}")

    # Embed
    texts = [(r.get("raw_text") or r.get("text") or "").strip() for r in records]
    logger.info("Starting Embedding...")
    embeddings = embed_batch(texts, embedding_model, batch_size=args.batch_size)

    logger.info(f"DEBUG: Valid Records = {len(records)}")
    logger.info(f"DEBUG: Generated Embeddings = {len(embeddings)}")

    # Index
    indexing(records, embeddings, milvus_client, args.collection)


if __name__ == "__main__":
    main()
