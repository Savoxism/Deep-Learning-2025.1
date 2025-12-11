import json
import numpy as np

def compute_hit_k(pred_items, gt_items, k = 5):
    # Lấy top k dự đoán và chuyển thành set để tìm giao
    top_k_set = set(pred_items[:k])
    gt_set = set(gt_items)
    
    # Nếu có giao nhau -> Hit
    return 1 if top_k_set.intersection(gt_set) else 0

def compute_mrr_k(pred_items, gt_items, k = 5):
    for rank, item in enumerate(pred_items[:k], start = 1):
        if item in gt_items:
            return 1.0 / rank
    return 0.0

if __name__ == "__main__":
    hits, mrrs = [], []
    k = 5
    
    INPUT_FILE = "results.json"

    print(f"📊 Loading data from {INPUT_FILE}...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        # SỬA Ở ĐÂY: Đọc toàn bộ mảng JSON thay vì đọc từng dòng
        data_list = json.load(f) 

        for data in data_list:
            # 1. Lấy list CID đúng (Ground Truth)
            gt_list = data['ground_truth_cids']
            
            # 2. Lấy list CID dự đoán
            pred_list = [item['cid'] for item in data['retrieved_top_k']]
            
            # 3. Tính điểm
            hits.append(compute_hit_k(pred_list, gt_list, k))
            mrrs.append(compute_mrr_k(pred_list, gt_list, k))

    if hits:
        print(f"Result @ Top {k} (on {len(hits)} queries):")
        print(f"🎯 Hit@{k}: {np.mean(hits):.4f}")
        print(f"🥇 MRR@{k}: {np.mean(mrrs):.4f}")
    else:
        print("⚠️ No data found to evaluate.")