import pandas as pd
import sys
import os
import argparse
import logging
import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from pymilvus import MilvusClient
from utils.models.embedding_model import EmbeddingModel
from chunking import SimpleChunker

from utils.helper import setup_logger

"""
python index.py \
  --input "data/filtered_corpus.csv" \
  --output "vector_db/soict_legal.db" \
  --collection "chunks_e5_base" \
  --model-name "intfloat/multilingual-e5-base" \
  --batch-size 16 \
"""

logger = setup_logger(
    log_file = "indexing.log"
)

def embed_batch(texts: List[str],
                embedding_model: EmbeddingModel,
                batch_size: int = 32) -> List:
    """
    Create embeddings for a batch of texts.
    """

    all_embeddings = []
    PROCESSING_BATCH = 2048

    total_steps = (len(texts) + PROCESSING_BATCH - 1) // PROCESSING_BATCH

    for i in tqdm(range(total_steps), desc = "Encoding Progress", total = total_steps):
        # Lấy một đoạn văn bản
        batch_texts = texts[i : i + PROCESSING_BATCH]

        try:
            batch_emb = embedding_model.encode_documents(
                batch_texts, 
                batch_size=batch_size 
            )

            all_embeddings.extend(list(batch_emb))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error encoding processing batch {i}: {str(e)}")
            # Fallback zero vector
            zeros = [np.zeros(768) for _ in range(len(batch_texts))]
            all_embeddings.extend(zeros)
            
    return all_embeddings

def indexing(chunks: List[Dict], embeddings: List, milvus_client: MilvusClient, collection_name: str):
    logger.info("="*40)
    logger.info(f"SAVING TO MILVUS: {collection_name}")

    # 1. Drop existing
    if milvus_client.has_collection(collection_name):
        logger.warning(f"Collection '{collection_name}' exists. Dropping...")
        milvus_client.drop_collection(collection_name)
    
    # 2. Create Collection
    if not embeddings:
        return
    
    dim = len(embeddings[0])
    logger.info(f"Embedding dimension: {dim}")
    
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="COSINE", 
        auto_id=True, 
        enable_dynamic_field=True
    )
    
    # 3. Prepare Data
    data = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        record = {
            "vector": emb, # Milvus Client tự xử lý numpy array
            "text": chunk['text'],
            "cid": str(chunk['cid']),     # Metadata: CID
            "chunk_index": chunk['chunk_index'],
            "word_count": chunk['word_count']
        }
        data.append(record)
        
    # 4. Insert Batch
    logger.info(f"Inserting {len(data)} records...")
    insert_batch = 1000
    for i in tqdm(range(0, len(data), insert_batch), desc="Inserting to DB"):
        milvus_client.insert(
            collection_name=collection_name,
            data=data[i : i + insert_batch]
        )

    logger.info("✅ Save complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/filtered_corpus.csv", help="CSV file")
    parser.add_argument("--output", default="vector_db/legal_race.db", help="Milvus DB Path")
    parser.add_argument("--collection", default="legal_chunks_e5", help="Collection Name")
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU Inference Batch Size")
    
    args = parser.parse_args()
    
    logger.info(f"START INDEXING: {args.input}")

    DB_DIR = "vector_db"
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    # load Data
    if not os.path.exists(args.input):
        logger.error(f"File missing: {args.input}")
        return
    df = pd.read_csv(args.input)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # initialuze
    chunker = SimpleChunker(chunk_size=256, overlap=32)
    embedding_model = EmbeddingModel(model_name=args.model_name, device = DEVICE)
    milvus_client = MilvusClient(args.output)

    # chunking
    all_chunks = []
    logger.info("Starting Chunking...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking Docs"):
        text_str = str(row['text']).strip()
        if not text_str: continue # Bỏ qua nếu text gốc rỗng
        
        chunks = chunker.chunk_text(text_str, row['cid'])
        all_chunks.extend(chunks)

    # === THÊM ĐOẠN NÀY ĐỂ LỌC LẦN CUỐI ===
    valid_chunks = []
    for c in all_chunks:
        if c['text'] and len(str(c['text']).strip()) > 0:
            valid_chunks.append(c)
    
    all_chunks = valid_chunks
    logger.info(f"Total valid chunks to embed: {len(all_chunks)}")

    # Embedding
    logger.info("Starting Embedding...")
    texts = [c['text'] for c in all_chunks]
    
    embeddings = embed_batch(
        texts, 
        embedding_model, 
        batch_size=args.batch_size
    )

    # save
    if len(embeddings) == len(all_chunks):
        indexing(all_chunks, embeddings, milvus_client, args.collection)
    else:
        logger.error("Mismatch count chunks vs embeddings!")

if __name__ == "__main__":
    main()