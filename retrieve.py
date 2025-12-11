import pandas as pd
import json
import ast
import os
import sys
from tqdm import tqdm

from pymilvus import MilvusClient
from utils.models.embedding_model import EmbeddingModel

DB_PATH = "vector_db/soict_legal.db"
COLLECTION_NAME = "chunks_e5_base"
INPUT_CSV = "data/processed_train.csv"
OUTPUT_FILE = "retrieval_results.jsonl"
TOP_K = 5

df = pd.read_csv(INPUT_CSV)
df_subset = df.head(100)

client = MilvusClient(uri=DB_PATH)
model = EmbeddingModel(model_name="intfloat/multilingual-e5-base")

results_to_save = []
print(f"🔄 Đang xử lý {len(df_subset)} câu hỏi...")

# PROCESSING
for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
    question = row['question']
    
    # obtain ground truth cids
    if isinstance(row['cid_list'], str):
        gt_cids = ast.literal_eval(row['cid_list'])
    else:
        gt_cids = row['cid_list']
        
    gt_cids = [str(c) for c in gt_cids]

    # embedding
    query_vec = model.encode_queries([question], batch_size=1)[0]
    
    # search
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=TOP_K,
        output_fields=["cid", "text"], 
        search_params={"metric_type": "COSINE", "params": {}}
    )
    
    # parse results
    retrieved_items = []
    for hit in search_res[0]:
        retrieved_items.append({
            "cid": str(hit['entity']['cid']),
            "score": float(hit['distance']),
            "text_snippet": hit['entity']['text'][:150] 
        })
        
    # append to results
    record = {
        "qid": row.get('qid', idx),
        "question": question,
        "ground_truth_cids": gt_cids,
        "retrieved_top_k": retrieved_items
    }
    results_to_save.append(record)


print(f"💾 Đang lưu kết quả vào {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for item in results_to_save:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
print("✅ Hoàn tất lưu file!")

