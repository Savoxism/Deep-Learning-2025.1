import os
import gc
import uuid
import torch
import asyncio
import logging
from types import SimpleNamespace

# Telegram imports
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode

# DB & Model imports
from pymilvus import MilvusClient
import numpy as np

from utils.models.vlm import VisionLanguageModel, VLMConfig
from utils.models.slm import LegalSLM, SLMConfig
from utils.models.embedder import EmbeddingModel
from utils.models.reranker import RerankingModel
from chunking import SimpleChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from huggingface_hub import login

HUGGINGFACE_API_TOKEN="<API_KEY>" # replace with your token
login(HUGGINGFACE_API_TOKEN)

TELEGRAM_TOKEN = "<API_KEY>" # replace with your token

# Model Configs
VLM_BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_ADAPTER = "Ewengc21/qwen_qlora_dl_project"
SLM_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct" # this must change. 
# SLM_ADAPTER = "Savoxism/Llama3-Adapter-DL-Project"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-base"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"

# Vector Database
DB_URI = "vector_db/milvus_demo.db"
COLLECTION_NAME = "legal_rag_collection"

# ---------------------------------------------------------
# 2. Global State & Lazy Loading
# ---------------------------------------------------------
print(">>> Initializing Core Components...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Lightweight Models Globally
try:
    embedder = EmbeddingModel(model_name=EMBEDDING_MODEL_ID, device=device)
except Exception as e:
    logger.error(f"Error loading embedder: {e}")
    embedder = None

try:
    reranker = RerankingModel(model_name=RERANKER_MODEL_ID)
except Exception as e:
    logger.error(f"Error loading reranker: {e}")
    reranker = None

chunker = SimpleChunker(chunk_size=256, overlap=32)
milvus_client = MilvusClient(uri=DB_URI)

# Heavy Models (Lazy Loaded)
vlm_model = None
slm_model = None

def clean_memory():
    """Aggressively clear GPU memory."""
    global vlm_model, slm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def load_vlm_lazy():
    global vlm_model, slm_model
    if slm_model is not None:
        logger.info("Unloading SLM to free VRAM...")
        del slm_model
        slm_model = None
        clean_memory()
    
    if vlm_model is None:
        logger.info(f"🔄 Lazy Loading VLM: {VLM_BASE_MODEL}...")
        conf = VLMConfig(
            base_model=VLM_BASE_MODEL, 
            adapter_model=VLM_ADAPTER,
            default_dpi=200,
            load_in_4bit=True
        )
        vlm_model = VisionLanguageModel(config=conf)
    return vlm_model

def load_slm_lazy():
    global vlm_model, slm_model
    if vlm_model is not None:
        logger.info("Unloading VLM to free VRAM...")
        del vlm_model
        vlm_model = None
        clean_memory()

    if slm_model is None:
        logger.info(f"🔄 Lazy Loading SLM: {SLM_BASE_MODEL}...")
        conf = SLMConfig(
            base_model=SLM_BASE_MODEL,
            # adapter_model=SLM_ADAPTER,
            load_in_4bit=True,
            max_seq_length=2048
        )
        slm_model = LegalSLM(config=conf)
    return slm_model

# ---------------------------------------------------------
# 3. Core Logic (Synchronous)
# ---------------------------------------------------------
# Note: These functions remain synchronous (blocking).
# We will run them in a separate thread using asyncio loop.run_in_executor

def process_pdf_ingestion(file_path: str, original_filename: str) -> str:
    # 1. Load VLM
    try:
        model = load_vlm_lazy()
    except Exception as e:
        return f"❌ Error loading VLM: {e}"

    output_md_path = file_path.replace(".pdf", ".md")
    
    # 2. Convert PDF to Markdown
    try:
        logger.info(f"Processing PDF: {file_path}")
        model.pdf_to_markdown(pdf_path=file_path, output_md_path=output_md_path, verbose=True)
        with open(output_md_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except Exception as e:
        return f"❌ Error during VLM Inference: {e}"

    # 3. Chunk & Embed
    doc_id = int(uuid.uuid4().int & (1<<32)-1)
    chunks = chunker.chunk_text(full_text, cid=doc_id)
    
    if not chunks:
        return "❌ Error: No text extracted."

    texts = [c['text'] for c in chunks]
    embeddings = embedder.encode(texts, batch_size=8) 

    # 4. Save to Milvus
    if not milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=embeddings.shape[1],
            metric_type="COSINE",
            auto_id=True,
            enable_dynamic_field=True
        )
    
    data = []
    for c, emb in zip(chunks, embeddings):
        vec = emb.tolist() if isinstance(emb, np.ndarray) else emb
        data.append({
            "vector": vec, 
            "text": c['text'], 
            "cid": str(c['cid']),
            "source": original_filename
        })

    milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
    
    # 5. Cleanup
    global vlm_model
    del vlm_model
    vlm_model = None
    clean_memory()

    return f"✅ **Success!**\nIndexed {len(chunks)} chunks from `{original_filename}`."

def rag_response(query: str, top_k: int=10, top_m: int=3):
    if not query.strip(): return "Empty query.", ""

    # 1. Load SLM
    try:
        model = load_slm_lazy()
    except Exception as e:
        return f"❌ Error loading SLM: {e}", ""

    # 2. Retrieve
    query_vec = embedder.encode([query], batch_size=1)[0]
    res = milvus_client.search(
        collection_name=COLLECTION_NAME, 
        data=[query_vec], 
        limit=top_k, 
        output_fields=["text", "source"]
    )
    
    if not res or not res[0]: 
        return "No relevant documents found.", ""
    
    hits = res[0]
    
    # 3. Rerank
    pairs = [[query, h['entity']['text']] for h in hits]
    scores = reranker.predict(pairs)
    scored_hits = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:top_m]
    
    # Context
    context_text = "\n\n".join([f"Source ({h['entity'].get('source', 'Unknown')} - Score {s:.2f}):\n{h['entity']['text']}" for h, s in scored_hits])
    
    # 4. Generate
    try:
        answer = model.generate(context=context_text, question=query)
    except Exception as e:
        answer = f"Error generating answer: {e}"

    return answer, context_text

# ---------------------------------------------------------
# 4. Telegram Bot Handlers
# ---------------------------------------------------------
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(
        "⚖️ **Legal AI Agent Ready**\n\n"
        "1. **Upload a PDF** to ingest and index a contract.\n"
        "2. **Type a question** to analyze indexed documents.\n\n"
        "*Note: Processing large PDFs takes time due to VLM OCR.*",
    )

@dp.message(F.document)
async def handle_document(message: Message):
    """
    Handles PDF uploads.
    """
    doc = message.document
    if not doc.file_name.lower().endswith('.pdf'):
        await message.answer("⚠️ Please upload a **PDF** file.")
        return

    status_msg = await message.answer(f"📥 Downloading `{doc.file_name}`...")
    
    # Create temp directory
    os.makedirs("temp_downloads", exist_ok=True)
    local_path = os.path.join("temp_downloads", doc.file_name)
    
    # Download file
    await bot.download(doc, destination=local_path)
    
    await status_msg.edit_text(f"👁️ **Processing VLM...**\nReading: `{doc.file_name}`\n_This uses GPU and may take a moment._")
    
    # Run blocking ingestion in a separate thread
    loop = asyncio.get_event_loop()
    try:
        # We wrap the call in an executor so it doesn't freeze the bot
        result = await loop.run_in_executor(
            None, 
            process_pdf_ingestion, 
            local_path, 
            doc.file_name
        )
        await status_msg.edit_text(result)
    except Exception as e:
        await status_msg.edit_text(f"❌ Critical Error: {str(e)}")
    finally:
        # Cleanup temp file
        if os.path.exists(local_path):
            os.remove(local_path)

@dp.message(F.text)
async def handle_text(message: Message):
    """
    Handles text queries (RAG).
    """
    query = message.text
    status_msg = await message.answer("🧠 **Thinking...** (Retrieving & Reasoning)")
    
    loop = asyncio.get_event_loop()
    try:
        # Run blocking RAG in executor
        answer, context = await loop.run_in_executor(
            None, 
            rag_response, 
            query, 
            10, # Top-K
            3   # Top-M
        )
        
        # Split message if too long for Telegram (Limit is 4096 chars)
        response_text = f"🤖 **AI Assessment:**\n{answer}"
        
        if len(response_text) > 4000:
            response_text = response_text[:4000]
            
        await status_msg.edit_text(response_text)
        
        # Send context as a separate message or file if requested
        # For this demo, we send a short snippet
        if context:
            # Create a small preview of context
            context_preview = f"📚 **Cited Evidence:**\n\n{context[:100]}"
            if len(context) > 1000: context_preview += "\n...(truncated)"
            await message.answer(context_preview)
            
    except Exception as e:
        await status_msg.edit_text(f"❌ Error during analysis: {str(e)}")

async def main() -> None:
    # Delete webhook if exists to ensure polling works
    await bot.delete_webhook(drop_pending_updates=True)
    print("🚀 Telegram Bot Started via Long Polling...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped.")