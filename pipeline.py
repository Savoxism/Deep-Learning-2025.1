import os
import gc
import uuid
import torch
import gradio as gr
from pymilvus import MilvusClient
import numpy as np

from utils.models.vlm import VisionLanguageModel, VLMConfig
from utils.models.slm import LegalSLM, SLMConfig
from utils.models.embedder import EmbeddingModel
from utils.models.reranker import RerankingModel
from chunking import SimpleChunker

# VLM Configuration
VLM_BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_ADAPTER = "Ewengc21/qwen_qlora_dl_project"

# SLM Configuration
SLM_BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"
SLM_ADAPTER = "Savoxism/Llama3-Adapter-DL-Project"

# Retrieval Models
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-base"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"

# Vector Database
DB_URI = "vector_db/milvus_demo.db"
COLLECTION_NAME = "legal_rag_collection"

# ---------------------------------------------------------
# 2. Global State & Lazy Loading
# ---------------------------------------------------------
# Lightweight models can be loaded globally or lazily depending on VRAM
# Here we load Embedder/Reranker globally as they are relatively small, 
# but SLM and VLM are swapped in/out.

print(">>> Initializing Core Components...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Embedder
try:
    embedder = EmbeddingModel(model_name=EMBEDDING_MODEL_ID, device=device)
except Exception as e:
    print(f"Error loading embedder: {e}")
    embedder = None

# Load Reranker
try:
    reranker = RerankingModel(model_name=RERANKER_MODEL_ID)
except Exception as e:
    print(f"Error loading reranker: {e}")
    reranker = None

# Initialize Chunker & DB
chunker = SimpleChunker(chunk_size=256, overlap=32)
milvus_client = MilvusClient(uri=DB_URI)

# Placeholders for Heavy Models
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
    """Loads VLM, unloading SLM if necessary."""
    global vlm_model, slm_model
    
    if slm_model is not None:
        print("Creating VRAM space: Unloading SLM...")
        del slm_model
        slm_model = None
        clean_memory()
    
    if vlm_model is None:
        print(f"🔄 Lazy Loading VLM: {VLM_BASE_MODEL}...")
        conf = VLMConfig(
            base_model=VLM_BASE_MODEL, 
            adapter_model=VLM_ADAPTER,
            default_dpi=200,
            load_in_4bit=True
        )
        vlm_model = VisionLanguageModel(config=conf)
    return vlm_model

def load_slm_lazy():
    """Loads SLM, unloading VLM if necessary."""
    global vlm_model, slm_model

    if vlm_model is not None:
        print("Creating VRAM space: Unloading VLM...")
        del vlm_model
        vlm_model = None
        clean_memory()

    if slm_model is None:
        print(f"🔄 Lazy Loading SLM: {SLM_BASE_MODEL}...")
        conf = SLMConfig(
            base_model=SLM_BASE_MODEL,
            adapter_model=SLM_ADAPTER,
            load_in_4bit=True,
            max_seq_length=2048
        )
        slm_model = LegalSLM(config=conf)
    return slm_model

# ---------------------------------------------------------
# 3. Core Logic
# ---------------------------------------------------------

def process_pdf_ingestion(pdf_file) -> str:
    """
    Pipeline: PDF -> VLM (Markdown) -> Chunker -> Embedder -> Milvus
    """
    # 1. Load VLM
    try:
        model = load_vlm_lazy()
    except Exception as e:
        return f"Error loading VLM (OOM?): {e}"

    if pdf_file is None: return "Error: No file uploaded."
    
    file_path = pdf_file.name
    output_md_path = file_path.replace(".pdf", ".md")
    
    # 2. Convert PDF to Markdown
    try:
        print(f"Processing PDF: {file_path}...")
        model.pdf_to_markdown(pdf_path=file_path, output_md_path=output_md_path, verbose=True)
        with open(output_md_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except Exception as e:
        return f"Error during VLM Inference: {e}"

    # 3. Chunk & Embed
    doc_id = int(uuid.uuid4().int & (1<<32)-1)
    chunks = chunker.chunk_text(full_text, cid=doc_id)
    print(f"Generated {len(chunks)} chunks.")
    
    if not chunks:
        return "Error: No text extracted."

    texts = [c['text'] for c in chunks]
    # Use smaller batch size to be safe with VRAM
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
        # Convert numpy array to list for JSON serialization
        vec = emb.tolist() if isinstance(emb, np.ndarray) else emb
        data.append({
            "vector": vec, 
            "text": c['text'], 
            "cid": str(c['cid']),
            "chunk_index": c['chunk_index']
        })

    milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
    
    # 5. Cleanup
    # Unload VLM immediately to free memory for Chat
    global vlm_model
    del vlm_model
    vlm_model = None
    clean_memory()
    print("🧹 Cleanup VLM finished.")

    return f"Success! Extracted and indexed {len(chunks)} chunks from '{os.path.basename(file_path)}'."

def rag_response(query: str, top_k: int, top_m: int):
    """
    Pipeline: Query -> Embedder -> Milvus (Top-K) -> Reranker (Top-M) -> SLM -> Answer
    """
    if not query.strip():
        return "Please enter a query.", ""

    # 1. Load SLM (Lazy)
    try:
        model = load_slm_lazy()
    except Exception as e:
        return f"Error loading SLM: {e}", ""

    # 2. Retrieve (Vector Search)
    query_vec = embedder.encode([query], batch_size=1)[0]
    res = milvus_client.search(
        collection_name=COLLECTION_NAME, 
        data=[query_vec], 
        limit=top_k, 
        output_fields=["text", "cid"]
    )
    
    if not res or not res[0]: 
        return "No relevant documents found in the database.", ""
    
    hits = res[0]
    
    # 3. Rerank
    # Prepare pairs [Query, Doc]
    pairs = [[query, h['entity']['text']] for h in hits]
    scores = reranker.predict(pairs)
    
    # Zip, Sort, and Slice Top-M
    scored_hits = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:top_m]
    
    # Build Context
    context_text = "\n\n".join([f"Document Fragment (Score {s:.4f}):\n{h['entity']['text']}" for h, s in scored_hits])
    
    # 4. Generate Answer
    try:
        answer = model.generate(context=context_text, question=query)
    except Exception as e:
        answer = f"Error generating answer: {e}"

    return answer, context_text

# ---------------------------------------------------------
# 4. Gradio Interface
# ---------------------------------------------------------

# Custom CSS for Professional Look
custom_css = """
.gradio-container {max_width: 1400px !important; margin: auto;}
.header-container {
    text-align: center; 
    margin-bottom: 2rem; 
    padding: 2rem; 
    background: linear-gradient(to right, #1e293b, #334155); 
    color: white; 
    border-radius: 12px;
}
.header-title {font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;}
.header-subtitle {font-size: 1rem; opacity: 0.8; font-family: monospace;}
.context-box {
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.6;
}
.terminal-log textarea {
    background-color: #0f172a !important;
    color: #4ade80 !important;
    font-family: 'Courier New', monospace !important;
}
.disclaimer {
    text-align: center; 
    font-size: 0.8rem; 
    color: #64748b; 
    margin-top: 2rem;
}
"""

legal_theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="blue",
).set(
    button_primary_background_fill="#2563eb",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(theme=legal_theme, css=custom_css, title="Legal AI Workbench") as demo:
    
    # --- Header ---
    with gr.Column(elem_classes="header-container"):
        gr.HTML(f"""
            <div class='header-title'>⚖️ Agentic Document Intelligence</div>
            <div class='header-subtitle'>
                VLM: {VLM_BASE_MODEL} | SLM: {SLM_BASE_MODEL} | Reranker: BGE-M3
            </div>
        """)

    # --- Tab 1: Ingestion ---
    with gr.Tab("📁 Document Ingestion"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Contract")
                gr.Markdown("Upload a PDF to extract structure, OCR content, and index it into the Vector Database.")
                
                pdf_input = gr.File(
                    label="Legal Document (PDF)", 
                    file_types=[".pdf"],
                    file_count="single",
                    height=250
                )
                
                process_btn = gr.Button("🚀 Process & Index Document", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 2. System Logs")
                status_output = gr.Textbox(
                    label="Execution Log", 
                    placeholder="Waiting for upload...",
                    lines=16, 
                    interactive=False,
                    elem_classes="terminal-log"
                )

    # --- Tab 2: Chat & Retrieval ---
    with gr.Tab("💬 Legal Analysis"):
        
        with gr.Group():
            with gr.Row(variant="panel"):
                query_input = gr.Textbox(
                    label="Legal Query",
                    placeholder="e.g., Under what conditions can the supplier terminate the agreement without notice?",
                    scale=4,
                    autofocus=True
                )
                ask_btn = gr.Button("Analyze", variant="primary", scale=1, size="lg")
        
        with gr.Accordion("⚙️ Retrieval Settings (Advanced)", open=False):
            with gr.Row():
                k_slider = gr.Slider(minimum=5, maximum=50, value=10, step=1, label="Retrieval (Top-K)")
                m_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Reranker (Top-M)")

        with gr.Row(equal_height=True):
            with gr.Column(scale=3, variant="panel"):
                gr.Markdown("### 🤖 AI Assessment")
                answer_output = gr.Markdown(
                    value="*The legal analysis will appear here...*",
                    line_breaks=True
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📚 Cited Evidence")
                context_output = gr.Textbox(
                    label="Retrieved Context (Raw)",
                    placeholder="No context retrieved yet.",
                    lines=20,
                    interactive=False,
                    elem_classes="context-box",
                    show_copy_button=True
                )

    # --- Footer ---
    gr.HTML("""
        <div class="disclaimer">
            ⚠️ <b>Disclaimer:</b> This AI tool is designed for assistance purposes only and does not constitute professional legal advice. 
            Always verify citations against original documents.
        </div>
    """)

    # --- Events ---
    process_btn.click(
        fn=process_pdf_ingestion, 
        inputs=[pdf_input], 
        outputs=[status_output]
    )

    ask_btn.click(
        fn=rag_response, 
        inputs=[query_input, k_slider, m_slider], 
        outputs=[answer_output, context_output]
    )
    query_input.submit(
        fn=rag_response, 
        inputs=[query_input, k_slider, m_slider], 
        outputs=[answer_output, context_output]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True, debug=True)