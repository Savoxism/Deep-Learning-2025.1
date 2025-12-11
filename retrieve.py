import pandas as pd
import json
import ast
import os
import sys
from tqdm import tqdm

from pymilvus import MilvusClient
from utils.models.embedder import EmbeddingModel
from utils.models.reranker import RerankingModel


DB_PATH = "vector_db/viettel_law.db"
COLLECTION_NAME = "chunks_e5_base"
INPUT_CSV = "data/processed_train.csv"
OUTPUT_FILE = "results.json" 
RETRIEVAL_TOP_K = 15  
RERANK_TOP_M = 5      

print("🚀 Loading Models & Data...")
df = pd.read_csv(INPUT_CSV)
df_subset = df.head(500) 

client = MilvusClient(uri=DB_PATH)
embedder = EmbeddingModel(model_name="intfloat/multilingual-e5-base")
reranker = RerankingModel(model_name="BAAI/bge-reranker-base")

results_to_save = []
print(f"🔄 Processing {len(df_subset)} queries with 2-Stage Retrieval...")

# PROCESSING LOOP
for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
    question = row['question']
    
    # Parse Ground Truth CIDs
    try:
        raw_cid = row.get('cid_list', row.get('cid'))
        if isinstance(raw_cid, str):
            gt_cids = ast.literal_eval(raw_cid)
        else:
            gt_cids = raw_cid
        gt_cids = [str(c) for c in gt_cids] # Normalize to string
    except:
        gt_cids = []

    # STAGE 1: VECTOR RETRIEVAL (Top-15)
    query_vec = embedder.encode_queries([question], batch_size=1)[0]
    
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=RETRIEVAL_TOP_K, # Lấy 15
        output_fields=["cid", "text"], 
        search_params={"metric_type": "COSINE", "params": {}}
    )

    # Prepare input for Reranker
    stage1_candidates = []
    pairs_to_rerank = []
    
    for hit in search_res[0]:
        doc_text = hit['entity']['text']
        # Lưu lại thông tin để dùng sau khi rerank
        candidate_info = {
            "cid": str(hit['entity']['cid']),
            "text": doc_text,
            "vector_score": float(hit['distance'])
        }
        stage1_candidates.append(candidate_info)
        # Tạo cặp [Query, Document]
        pairs_to_rerank.append([question, doc_text])

    # ==========================
    # STAGE 2: RERANKING (Top-5)
    if pairs_to_rerank:
        # Predict scores
        rerank_scores = reranker.predict(pairs_to_rerank)
        
        # Gán điểm rerank vào candidate
        for i, score in enumerate(rerank_scores):
            stage1_candidates[i]['rerank_score'] = score
        
        # Sắp xếp lại theo điểm Rerank (Cao -> Thấp)
        stage1_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # Cắt lấy Top-M (Top 5)
    final_top_results = stage1_candidates[:RERANK_TOP_M]

    # FORMAT RESULT
    formatted_retrieved = []
    for item in final_top_results:
        formatted_retrieved.append({
            "cid": item['cid'],
            "rerank_score": round(item['rerank_score'], 4),
            "vector_score": round(item['vector_score'], 4),
            "text_snippet": item['text'][:].replace('\n', ' ') + "..."
        })

    record = {
        "qid": row.get('qid', idx),
        "question": question,
        "ground_truth_cids": gt_cids,
        "retrieved_top_k": formatted_retrieved
    }
    results_to_save.append(record)


print(f"💾 Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results_to_save, f, ensure_ascii=False, indent=4)
    
print("✅ Done! Check the output file.")
