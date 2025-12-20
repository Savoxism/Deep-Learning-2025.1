import os
import gradio as gr
import torch
import numpy as np
import uuid
from typing import List, Tuple
from pymilvus import MilvusClient

# --- Import local modules ---
from utils.models.vlm import VisionLanguageModel, VLMConfig
from utils.models.embedder import EmbeddingModel
from utils.models.reranker import RerankingModel
from utils.models.slm import LegalSLM, SLMConfig
from chunking import SimpleChunker

# Models
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct" 

EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-base" 
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"

SLM_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SLM_ADAPTER_PATH = None

DB_URI = "vector_db/milvus_demo.db"
COLLECTION_NAME = "demo_rag_collection"

# --- Global Model Initialization ---
print(">>> Initializing Models...")

# 1. Initialize VLM (Vision Language Model)
vlm_config = VLMConfig(
    base_model=VLM_MODEL_ID,
    load_in_4bit = True,
    adapter_model=None,
    device_map="auto", 
    default_max_new_tokens=1024,
    default_dpi=200,
)

try:
    print(f"Loading VLM: {VLM_MODEL_ID}...")
    vlm_model = VisionLanguageModel(config=vlm_config)
except Exception as e:
    print(f"⚠️ Warning: VLM failed to load (likely OOM). Error: {e}")
    vlm_model = None

# 2. Initialize Embedder
print(f"Loading Embedder: {EMBEDDING_MODEL_ID}...")
embedder = EmbeddingModel(model_name=EMBEDDING_MODEL_ID, device="cuda" if torch.cuda.is_available() else "cpu")

# 3. Initialize Reranker
print(f"Loading Reranker: {RERANKER_MODEL_ID}...")
reranker = RerankingModel(model_name=RERANKER_MODEL_ID)

# 4. Initialize SLM (Small Language Model) - Updated for new slm.py structure
print(f"Loading SLM: {SLM_BASE_MODEL}...")
try:
    # Use SLMConfig to configure the model
    slm_conf = SLMConfig(
        base_model=SLM_BASE_MODEL,
        adapter_model=SLM_ADAPTER_PATH, # None for now (Pure Qwen 3B)
        load_in_4bit=True,
        max_seq_length=2048
    )
    slm_model = LegalSLM(config=slm_conf)
except Exception as e:
    print(f"⚠️ Warning: SLM failed to load. Error: {e}")
    slm_model = None

# 5. Initialize Chunker
chunker = SimpleChunker(chunk_size=128, overlap=16)

# 6. Initialize Database
milvus_client = MilvusClient(uri=DB_URI)

print(">>> Initialization Complete.")


# --- Core Pipeline Functions ---
def process_pdf_ingestion(pdf_file) -> str:
    """
    Ingestion Pipeline:
    1. VLM: PDF -> Markdown
    2. Chunker: Text -> Chunks
    3. Embedder: Chunks -> Vectors
    4. Milvus: Store Data
    """
    if vlm_model is None:
        return "Error: VLM Model not loaded. Check GPU memory."
    
    if pdf_file is None:
        return "Error: No file uploaded."

    file_path = pdf_file.name
    print(f"Processing file: {file_path}")

    # Step 1: PDF to Markdown
    output_md_path = file_path.replace(".pdf", ".md")
    try:
        vlm_model.pdf_to_markdown(
            pdf_path=file_path,
            output_md_path=output_md_path,
            verbose=True
        )
        with open(output_md_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except Exception as e:
        return f"Error during VLM processing: {str(e)}"

    # Step 2: Chunking
    doc_id = int(uuid.uuid4().int & (1<<32)-1) 
    chunks = chunker.chunk_text(full_text, cid=doc_id)
    print(f"Generated {len(chunks)} chunks.")

    if not chunks:
        return "Error: No text extracted or chunked."

    # Step 3: Embedding
    texts_to_embed = [c['text'] for c in chunks]
    # Small batch size to save VRAM since we have many models loaded
    embeddings = embedder.encode(texts_to_embed, batch_size=8) 

    # Step 4: Indexing to Milvus
    if not milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=embeddings.shape[1],
            metric_type="COSINE",
            auto_id=True,
            enable_dynamic_field=True
        )

    data_to_insert = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        data_to_insert.append({
            "vector": emb.tolist(),
            "text": chunk['text'],
            "cid": str(chunk['cid']),
            "chunk_index": chunk['chunk_index']
        })

    milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)

    return f"Success! Processed '{os.path.basename(file_path)}'. \nExtracted {len(chunks)} chunks.\nSaved to Milvus collection: '{COLLECTION_NAME}'."


def rag_response(query: str, top_k: int = 10, top_m: int = 3):
    """
    Retrieval & Generation Pipeline:
    1. Embed Query
    2. Vector Search (Top-K)
    3. Rerank (Top-M)
    4. Generate Answer (SLM)
    """
    if not query.strip():
        return "Please enter a query.", ""

    if slm_model is None:
        return "Error: SLM Model not loaded.", ""

    # Step 1: Embed Query
    query_vec = embedder.encode([query], batch_size=1)[0]

    # Step 2: Retrieve Top-K
    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=top_k,
        output_fields=["text", "cid"],
        search_params={"metric_type": "COSINE", "params": {}}
    )

    if not search_res or not search_res[0]:
        return "No relevant documents found.", ""

    retrieved_items = search_res[0] 
    
    # Step 3: Reranking
    # Prepare pairs: [Query, Document]
    rerank_pairs = [[query, hit['entity']['text']] for hit in retrieved_items]
    scores = reranker.predict(rerank_pairs)
    
    # Combine content with scores
    scored_results = []
    for hit, score in zip(retrieved_items, scores):
        scored_results.append({
            "text": hit['entity']['text'],
            "score": score,
            "cid": hit['entity']['cid']
        })

    # Sort by score and take Top-M
    scored_results.sort(key=lambda x: x['score'], reverse=True)
    final_context_docs = scored_results[:top_m]

    # Construct Context String
    context_str = "\n\n".join([f"Document (Score: {doc['score']:.4f}):\n{doc['text']}" for doc in final_context_docs])

    # Step 4: Generate Answer
    try:
        # Note: Qwen 2.5 is very good at following instructions
        answer = slm_model.generate(context=context_str, question=query)
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return answer, context_str


# --- Gradio User Interface ---
custom_css = """
#component-0 {max_width: 1200px; margin: auto;}
.context-box {font-size: 12px; color: #555; font-family: monospace;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Legal RAG Pipeline (Qwen 3B Edition)") as demo:
    gr.Markdown("# 🤖 Legal AI RAG Pipeline")
    gr.Markdown(f"**Models:** VLM: {VLM_MODEL_ID} | SLM: {SLM_BASE_MODEL} | Reranker: {RERANKER_MODEL_ID}")

    with gr.Tab("📁 Step 1: Data Ingestion"):
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload Legal PDF", file_types=[".pdf"])
                process_btn = gr.Button("🚀 Process & Index", variant="primary")
            
            with gr.Column(scale=2):
                status_output = gr.Textbox(label="Processing Log", lines=10, interactive=False)

        process_btn.click(
            fn=process_pdf_ingestion,
            inputs=[pdf_input],
            outputs=[status_output]
        )

    with gr.Tab("💬 Step 2: Chat & Retrieval"):
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(label="Enter your legal query", placeholder="e.g., What are the penalties for late tax filing?")
                
                with gr.Row():
                    k_slider = gr.Slider(minimum=5, maximum=50, value=10, step=1, label="Retrieval Top-K")
                    m_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Reranker Top-M")
                
                ask_btn = gr.Button("🔍 Ask AI", variant="primary")
            
            with gr.Column(scale=3):
                answer_output = gr.Markdown(label="AI Answer")
                with gr.Accordion("📚 Retrieved Context (Reranked)", open=False):
                    context_output = gr.Textbox(label="Raw Context", lines=10, interactive=False, elem_classes="context-box")

        ask_btn.click(
            fn=rag_response,
            inputs=[query_input, k_slider, m_slider],
            outputs=[answer_output, context_output]
        )

if __name__ == "__main__":
    # Launch with share=True to create a public link
    demo.launch(share=True, debug=True)