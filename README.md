# Agentic Document Intelligence (ADI)
A Multi-Modal AI Agent for Automated Legal Contract Review & Reasoning

## Project Overview
Agentic Document Intelligence (ADI) is an advanced AI system designed to automate the extraction, analysis, and reasoning of complex legal contracts.

Traditional LegalTech tools often rely on fragile OCR pipelines and keyword matching, which fail when faced with complex layouts, embedded tables, or cross-referenced clauses. ADI overcomes this by leveraging a biologically inspired multi-model architecture. It combines Vision-Language Models (VLM) for pixel-perfect structural understanding with Small Language Models (SLM) specialized for high-precision legal reasoning.

The system is built to handle the nuances of legal documents—multi-column layouts, checkboxes, signatures, and hierarchical clauses—reducing the time legal professionals spend on manual review through an interactive, citation-backed Q&A interface.

The system operates through four distinct, cohesive modules that mimic cognitive processing:

## The Eye: Vision-Language Model (VLM)
Role: Structural Extraction & Layout Analysis

Model: `Qwen2.5-VL-3B-Instruct` (Quantized)

"The Eye" replaces traditional OCR. Instead of treating a PDF as a stream of characters, it sees the document as an image. It is able to recognize tables as structured data, identifies checked/unchecked checkboxes, and preserves header hierarchy. In addition, it ignores watermarks and scan noise that typically break OCR engines.

Output: Converts raw PDF pages into clean, structure-aware Markdown.

## The Memory: Retrieval-Augmented Generation (RAG)
Role: Context Indexing & Retrieval

Stack: Milvus (Vector DB), `multilingual-e5-base` (Embedding), `bge-reranker` (Reranking)

"The Memory" solves the limited context window of LLMs by creating a searchable index of the contract. For chunking, it uses a recursive strategy to keep legal clauses (Articles/Sections) intact rather than splitting them mid-sentence. As for retrieval, it combines semantic search (vector) with exact keyword matching.

Reranking: A Cross-Encoder (`bge-reranker`) scores the retrieved documents to ensure the most relevant clauses are prioritized before reaching the reasoning model.

## The Brain: Specialized Small Language Model (SLM)
Role: Reasoning, Risk Assessment & Entity Extraction

Model: `Llama-3` with LoRA Adapter.

"The Brain" is the cognitive core. It does not just summarize; it acts as a legal auditor. It has been finetuned to understand legal definitions, obligations, and liabilities. It could identify specific dates, monetary amounts, and parties involved. Furthermore, it can answer complex natural language queries while citing specific clauses from the contract.

## The Face: User Interface
Role: Interaction & Visualization

The interaction layer allows users to upload documents and query the agent in real-time. It provides a transparent view of the AI's reasoning process by displaying the retrieved context alongside the generated answer.

## Key Features
+ No-OCR Pipeline: Directly processes visual document layouts into Markdown, preserving tables and forms.

+ Intelligent Chunking: Breaks documents down by legal structure (Article -> Section -> Clause) rather than arbitrary word counts.

+ Reranked Retrieval: Uses a 2-stage retrieval process (Retrieve Top-K -> Rerank Top-M) to drastically improve accuracy.

+ Consumer Hardware Ready: Implements Lazy Loading resource management, allowing the entire pipeline (VLM + SLM + Reranker) to run on consumer GPUs (e.g., NVIDIA T4 or RTX 3060) by dynamically loading and unloading models.

+ Interactive QA: Users can ask questions like "What is the penalty for late delivery?" and receive precise, context-backed answers.