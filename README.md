# 📊 SEC Risk Analyzer – Local RAG System

## 🔎 Overview

A fully local **Retrieval-Augmented Generation (RAG)** system for analyzing SEC 10-K filings.

This project combines **semantic search (FAISS)** with a **locally hosted LLM (Ollama)** to generate grounded financial risk summaries — without using any external APIs.

---

## 🏗️ Architecture


Streamlit UI
↓
FAISS Vector Search
↓
BAAI/bge-base-en-v1.5 (GPU embeddings)
↓
Ollama (Mistral 7B)


---

## 🖥️ Application Screenshots

### 🔹 Main UI
![Main UI](screenshots/ui_home.png)

### 🔹 Sample Query Result
![Query Result](screenshots/query_result.png)

---

## ⚙️ Tech Stack

- Python  
- FAISS  
- Sentence Transformers  
- PyTorch (GPU-enabled)  
- Streamlit  
- Ollama  
- Mistral 7B (local LLM)

---

## ✨ Features

- GPU-accelerated embeddings  
- Semantic search over SEC filings  
- Context-grounded answer generation  
- Bullet-point financial risk summaries  
- Fully offline execution (no OpenAI / cloud APIs)  
- Modular RAG architecture  

---

## 🚀 How to Run

### 1️⃣ Create environment

```bash
conda create -n rag_env python=3.10
conda activate rag_env
pip install -r requirements.txt

### 2️⃣ Install Ollama & pull model
ollama pull mistral

### 3️⃣ Run the application
streamlit run app.py


## 📌 Example Query

What risks did Tesla disclose in 2023?

The system retrieves relevant sections from SEC filings and generates a grounded, bullet-point risk summary.

## 🧠 Future Improvements

Improved chunking strategy

Metadata filtering (company / year)

Hybrid search (BM25 + vector)

Reranking

Citation formatting

Answer confidence scoring

## 📄 Disclaimer

This project is for educational and research purposes only and does not constitute financial or investment advice.