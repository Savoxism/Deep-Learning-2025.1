from typing import List, Dict

class SimpleChunker:
    def __init__(self, chunk_size: int = 256, overlap: int = 32):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def chunk_text(self, text: str, cid: int) -> List[Dict]:
        """
        Cắt văn bản thành các đoạn nhỏ.
        """
        if not isinstance(text, str) or not text:
            return []

        # split by whitespace
        words = text.split()
        
        # If text is shorter than chunk_size, return as a single chunk
        if len(words) <= self.chunk_size:
            return [{
                'cid': cid,
                'text': text,
                'chunk_index': 0,
                'word_count': len(words)
            }]

        chunks = []
        chunk_idx = 0
        
        # sliding Window
        for i in range(0, len(words), self.step):
            chunk_words = words[i : i + self.chunk_size]
            
            if not chunk_words:
                continue

            chunk_str = " ".join(chunk_words)
            
            chunks.append({
                'cid': cid,
                'text': chunk_str,
                'chunk_index': chunk_idx,
                'word_count': len(chunk_words)
            })
            chunk_idx += 1

        return chunks