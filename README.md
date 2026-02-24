📊 SEC Risk Analyzer – Local RAG System
🔎 Overview

A fully local Retrieval-Augmented Generation (RAG) system for analyzing SEC 10-K filings.

This project combines semantic search (FAISS) with a locally hosted LLM to generate grounded financial risk summaries — without using any external APIs.

🏗️ Architecture

Streamlit UI
      ↓
FAISS Vector Search
      ↓
BAAI/bge-base-en-v1.5 (GPU embeddings)
      ↓
Ollama (Mistral 7B)

⚙️ Tech Stack

Python
FAISS
Sentence Transformers
PyTorch (GPU-enabled)
Streamlit
Ollama
Mistral 7B (local LLM)

✨ Features

GPU-accelerated embeddings
Semantic search over SEC filings
Context-grounded answer generation
Bullet-point financial risk summaries
Fully offline execution (no OpenAI API)
Expandable RAG architecture

🚀 How to Run
1️.Create environment
conda create -n rag_env python=3.10
conda activate rag_env
pip install -r requirements.txt

2️.Install Ollama & pull model
ollama pull mistral

3️.Run UI
streamlit run app.py

📌 Example Query

What risks did Tesla disclose in 2023?

The system retrieves relevant 10-K sections and generates a grounded summary.

🧠 Future Improvements

Improved chunking strategy
Metadata filtering (company/year)
Hybrid search (BM25 + vector)
Reranking
Citation formatting
Answer confidence scoring