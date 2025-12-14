from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Union

class EmbeddingModel:
    def __init__(self, model_name="intfloat/multilingual-e5-base", max_seq_length=512, device=None):
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
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,     
            normalize_embeddings=True  
        )
        return embeddings
