import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", max_length: int = 512, device: str = None):
        """
        Wrapper class cho E5 Embedding Model.
        Tự động xử lý prefix 'query:' và 'passage:' theo yêu cầu của E5 paper.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading model {model_name} on {self.device}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_length
        self.max_length = max_length

    def encode_queries(self, queries: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode câu hỏi (thêm prefix 'query: ')"""
        # E5 yêu cầu prefix 'query: ' cho câu hỏi
        processed_queries = [f"query: {q}" for q in queries]
        
        return self.model.encode(
            processed_queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    def encode_documents(self, documents: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode văn bản/corpus (thêm prefix 'passage: ')"""
        # E5 yêu cầu prefix 'passage: ' cho documents
        processed_docs = [f"passage: {d}" for d in documents]
        
        return self.model.encode(
            processed_docs,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        )