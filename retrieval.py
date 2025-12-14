import pandas as pd
import json
import ast
import re
from tqdm import tqdm

from pymilvus import MilvusClient
from utils.models.embedder import EmbeddingModel
from utils.models.reranker import RerankingModel

# --- CONFIG ---
DB_PATH = "vector_db/VN_legal.db" 
COLLECTION_NAME = "default" 
INPUT_CSV = "data/processed_train.csv"
OUTPUT_FILE = "outputs.json" 
RETRIEVAL_TOP_K = 15  
RERANK_TOP_M = 5      
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
RERANKING_MODEL_NAME = "BAAI/bge-reranker-base"


df = pd.read_csv(INPUT_CSV)
df_subset = df.head(100) # process some queries

client = MilvusClient(uri=DB_PATH)
print("using embedder model:", EMBEDDING_MODEL_NAME)
embedder = EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)
reranker = RerankingModel(model_name=RERANKING_MODEL_NAME)

results_to_save = []
# main processing loop
for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
    question = str(row['question'])
    
    # Parse ground truth cids
    gt_cids = []
    try:
        raw_cid = row.get('cid_list', row.get('cid'))
        if isinstance(raw_cid, str):
            raw_cid = raw_cid.strip()
            if raw_cid.startswith('[') and raw_cid.endswith(']'):
                gt_cids = ast.literal_eval(raw_cid)
            else:
                gt_cids = re.findall(r'\d+', raw_cid)
        elif isinstance(raw_cid, (list, tuple, int, float)):
            if isinstance(raw_cid, (int, float)):
                gt_cids = [raw_cid]
            else:
                gt_cids = raw_cid
        
        gt_cids = [str(c) for c in gt_cids]
    except Exception:
        gt_cids = []

    # STAGE 1: VECTOR RETRIEVAL (Top-15)
    try:
        query_vec = embedder.encode([question], batch_size=1)[0]

        search_res = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vec],
            limit=RETRIEVAL_TOP_K,
            output_fields=["cid", "text"], 
            search_params={"metric_type": "COSINE", "params": {}}
        )
    except Exception as e:
        print(f"❌ Error searching query {idx}: {e}")
        continue

    # Prepare input for Reranker
    stage1_candidates = []
    pairs_to_rerank = []
    
    if not search_res:
        continue

    for hit in search_res[0]:
        doc_text = hit['entity'].get('text', "")
        candidate_info = {
            "cid": str(hit['entity']['cid']),
            "text": doc_text,
            "vector_score": float(hit['distance'])
        }
        stage1_candidates.append(candidate_info)
        pairs_to_rerank.append([question, doc_text])

    # STAGE 2: RERANKING (Top-5)
    if pairs_to_rerank:
        try:
            rerank_scores = reranker.predict(pairs_to_rerank)
            
            for i, score in enumerate(rerank_scores):
                if i < len(stage1_candidates):
                    stage1_candidates[i]['rerank_score'] = score
            
            stage1_candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        except Exception as e:
            print(f"⚠️ Rerank error query {idx}: {e}")
            stage1_candidates.sort(key=lambda x: x['vector_score'], reverse=True)
    
    # top-m
    final_top_results = stage1_candidates[:RERANK_TOP_M]

    # formatting
    formatted_retrieved = []
    for item in final_top_results:
        formatted_retrieved.append({
            "cid": item['cid'],
            "rerank_score": round(item.get('rerank_score', 0.0), 4),
            "vector_score": round(item['vector_score'], 4),
            "text_snippet": item['text'].replace('\n', ' ') 
        })

    record = {
        "qid": row.get('qid', idx),
        "question": question,
        "ground_truth_cids": gt_cids,
        "retrieved_top_k": formatted_retrieved
    }
    results_to_save.append(record)


with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results_to_save, f, ensure_ascii=False, indent=4)
    
print("✅ Done! Check the output file.")
