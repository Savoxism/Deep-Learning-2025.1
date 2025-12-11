from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Union

class EmbeddingModel:
    def __init__(self, model_name="AITeamVN/Vietnamese_Embedding", max_seq_length=2048, device=None):
        """
        Wrapper fpr SentenceTransformer.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading Model: {model_name} on {self.device}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length

    def encode(self, texts: Union[str, List[str]], batch_size=32) -> np.ndarray:
        """
        Hàm encode chung cho cả document và query.
        Trả về Numpy Array (đã normalize) để tiết kiệm GPU RAM.
        """
        # Tự động xử lý batching và progress bar bên trong SentenceTransformer
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,     # Quan trọng: Chuyển về CPU ngay
            normalize_embeddings=True  # Quan trọng: Dùng cosine similarity cần cái này
        )
        return embeddings

    # Giữ lại các alias này để tương thích với code cũ của bạn
    def encode_query(self, query: Union[str, List[str]], batch_size=32) -> np.ndarray:
        return self.encode(query, batch_size=batch_size)

    def encode_documents(self, documents: Union[str, List[str]], batch_size=32) -> np.ndarray:
        # Với BGE-M3/Vietnamese_Embedding, thường không cần prefix "passage: "
        # trừ khi model card yêu cầu cụ thể.
        return self.encode(documents, batch_size=batch_size)

    def encode_queries(self, queries: List[str], batch_size=32) -> np.ndarray:
        return self.encode(queries, batch_size=batch_size)