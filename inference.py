import argparse
import sys
import os

from utils.models.slm import LegalSLM

def main():
    parser = argparse.ArgumentParser(description="Chạy Inference cho Legal RAG System")
    parser.add_argument("--model_path", type=str, default="models/llama3_legal_adapter", help="Đường dẫn đến folder model adapter")
    parser.add_argument("--query", type=str, default="Mức phạt nồng độ cồn tối đa là bao nhiêu?", help="Câu hỏi người dùng")
    parser.add_argument("--context", type=str, default=None, help="Ngữ cảnh (nếu không nhập sẽ dùng context giả lập)")
    
    args = parser.parse_args()

    # 2. Khởi tạo Model (Chỉ init 1 lần)
    # Lưu ý: Nếu chạy trong production server (như FastAPI), bước này nên để ở startup event
    try:
        bot = LegalSLM(model_path=args.model_path)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo model: {e}")
        return

    # 3. Giả lập Context (Nếu user không nhập)
    # Trong thực tế, context này đến từ Vector DB (Milvus) ở Part 2
    context = args.context
    if not context:
        print("\n⚠️ Không có context đầu vào, sử dụng Context giả lập (Mocking RAG retrieval)...")
        context = """
        Trích văn bản pháp luật:
        Điều 5. Xử phạt vi phạm quy định về nồng độ cồn
        3. Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng đối với người điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn vượt quá 80 miligam/100 mililít máu hoặc vượt quá 0,4 miligam/1 lít khí thở.
        """

    query = args.query

    # 4. Chạy sinh câu trả lời
    print("-" * 50)
    print(f"❓ Câu hỏi: {query}")
    print(f"📄 Context: {context.strip()[:100]}...") # In gọn context
    print("-" * 50)
    print("⏳ Đang suy nghĩ...")

    response = bot.generate(context=context, question=query)

    # 5. Kết quả
    print("\n🤖 TRẢ LỜI:")
    print("=" * 50)
    print(response)
    print("=" * 50)

if __name__ == "__main__":
    main()