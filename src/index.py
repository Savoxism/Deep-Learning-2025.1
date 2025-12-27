import pandas as pd
import os
import argparse
import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from pymilvus import MilvusClient
from utils.models.embedder import EmbeddingModel
from chunking import SimpleChunker

from src.helper import setup_logger

"""
python index.py --input "data/filtered_corpus.csv" --output "vector_db/VN_legal.db" --collection "default" --model-name "intfloat/multilingual-e5-base" --batch-size 64
"""

logger = setup_logger(log_file="indexing.log")

def embed_batch(texts: List[str], embedding_model: EmbeddingModel, batch_size: int = 32) -> List:
    """
    Create embeddings for a batch of texts.
    """
    all_embeddings = []
    PROCESSING_BATCH = 512

    # Lấy dimension mẫu từ model để phòng hờ trường hợp lỗi cần padding zero
    dummy_emb = embedding_model.encode(["test"], batch_size=1)
    EMBEDDING_DIM = dummy_emb.shape[1]
    logger.info(f"Detected Model Dimension: {EMBEDDING_DIM}")

    for i in tqdm(range(0, len(texts), PROCESSING_BATCH), desc="Encoding Progress"):
        batch_texts = texts[i : i + PROCESSING_BATCH]

        try:
            # Gọi model encode
            batch_emb = embedding_model.encode(
                batch_texts, 
                batch_size=batch_size 
            )
            
            # check for batch size mismatch
            if len(batch_emb) != len(batch_texts):
                logger.warning(f"⚠️ Batch mismatch at index {i}! Input: {len(batch_texts)}, Output: {len(batch_emb)}")
                missing_count = len(batch_texts) - len(batch_emb)
                
                # Dynamic Dimension Zero Padding
                zeros = [np.zeros(EMBEDDING_DIM) for _ in range(missing_count)]
                
                current_emb_list = list(batch_emb)
                batch_emb = current_emb_list + zeros

            all_embeddings.extend(list(batch_emb))

            # Clean GPU Cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"❌ Error encoding batch starting at {i}: {str(e)}")
            # Fallback toàn bộ batch bằng vector 0 với đúng dimension
            zeros = [np.zeros(EMBEDDING_DIM) for _ in range(len(batch_texts))]
            all_embeddings.extend(zeros)
            
    return all_embeddings

def indexing(chunks: List[Dict], embeddings: List, milvus_client: MilvusClient, collection_name: str):
    logger.info("="*40)
    logger.info(f"SAVING TO MILVUS: {collection_name}")

    # drop existing 
    if milvus_client.has_collection(collection_name):
        logger.warning(f"Collection '{collection_name}' exists. Dropping...")
        milvus_client.drop_collection(collection_name)
    
    # validate Embeddings
    if not embeddings:
        logger.error("No embeddings to save!")
        return
    
    # get dimension
    first_vec = embeddings[0]
    dim = len(first_vec)
    logger.info(f"Final Embedding dimension: {dim}")
    
    # create Collection
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="COSINE", 
        auto_id=True, 
        enable_dynamic_field=True
    )
    
    # 4. Prepare Data
    data = []
    logger.info(f"Preparing {len(chunks)} records...")
    
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        # Convert numpy array to list for Milvus (quan trọng)
        vec_list = emb.tolist() if isinstance(emb, np.ndarray) else emb
        
        record = {
            "vector": vec_list, 
            "text": chunk['text'],
            "cid": str(chunk['cid']),     
            "chunk_index": chunk['chunk_index'],
            "word_count": chunk['word_count']
        }
        data.append(record)
        
    # 5. Insert Batch
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
    parser.add_argument("--output", default="vector_db/default.db", help="Milvus DB Path")
    parser.add_argument("--collection", default="deep_learning_proj", help="Collection Name")
    # Default model name updated
    parser.add_argument("--model-name", default="AITeamVN/Vietnamese_Embedding")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU Inference Batch Size")
    
    args = parser.parse_args()
    
    logger.info(f"START INDEXING: {args.input}")
    logger.info(f"MODEL: {args.model_name}")

    # Create DB Directory
    DB_DIR = os.path.dirname(args.output) 
    if DB_DIR and not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    # Load Data
    df = pd.read_csv(args.input)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Components
    chunker = SimpleChunker(chunk_size=256, overlap=32)
    embedding_model = EmbeddingModel(
        model_name=args.model_name,
        max_seq_length = 512,
        device=DEVICE
    )
    milvus_client = MilvusClient(uri=args.output) 

    # Chunking
    all_chunks = []
    logger.info("Starting Chunking...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking Docs"):
        text_str = str(row['text']).strip()
        if not text_str: continue 
        
        chunks = chunker.chunk_text(text_str, row['cid'])
        all_chunks.extend(chunks)

    # Filter empty chunks
    valid_chunks = []
    for c in all_chunks:
        if c['text'] and len(str(c['text']).strip()) > 0:
            valid_chunks.append(c)
    
    all_chunks = valid_chunks
    logger.info(f"Total valid chunks to embed: {len(all_chunks)}")

    # 2. Embedding
    logger.info("Starting Embedding...")
    texts = [c['text'] for c in all_chunks]
    
    # Gọi hàm embed_batch (đã fix dimension)
    embeddings = embed_batch(
        texts, 
        embedding_model, 
        batch_size=args.batch_size
    )

    logger.info(f"DEBUG: Valid Chunks = {len(all_chunks)}")
    logger.info(f"DEBUG: Generated Embeddings = {len(embeddings)}")

    # 3. Save to Milvus
    if len(embeddings) == len(all_chunks):
        indexing(all_chunks, embeddings, milvus_client, args.collection)
    else:
        logger.error("Mismatch count chunks vs embeddings! Indexing aborted to prevent data corruption.")

if __name__ == "__main__":
    main()