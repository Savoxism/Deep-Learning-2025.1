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

        # Tách văn bản thành list các từ (dựa trên khoảng trắng)
        words = text.split()
        
        # Nếu văn bản ngắn hơn chunk_size, trả về nguyên văn
        if len(words) <= self.chunk_size:
            return [{
                'cid': cid,
                'text': text,
                'chunk_index': 0,
                'word_count': len(words)
            }]

        chunks = []
        chunk_idx = 0
        
        # Vòng lặp Sliding Window
        # range(start, stop, step)
        for i in range(0, len(words), self.step):
            # Lấy slice từ i đến i + chunk_size
            chunk_words = words[i : i + self.chunk_size]
            
            # Bỏ qua nếu chunk rỗng (trường hợp hiếm)
            if not chunk_words:
                continue

            # Ghép lại thành chuỗi
            chunk_str = " ".join(chunk_words)
            
            chunks.append({
                'cid': cid,
                'text': chunk_str,
                'chunk_index': chunk_idx,
                'word_count': len(chunk_words)
            })
            chunk_idx += 1

        return chunks