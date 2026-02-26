📊 SEC Risk Analyzer – Local RAG System
🔎 Overview

A fully local Retrieval-Augmented Generation (RAG) system for analyzing SEC 10-K filings.

This project combines semantic search using FAISS with a locally hosted large language model (via Ollama) to generate grounded, bullet-point financial risk summaries — without relying on any external APIs.

🏗️ Architecture

The system follows a simple RAG pipeline:

Streamlit-based user interface

FAISS vector search for semantic retrieval

BAAI/bge-base-en-v1.5 model for GPU-accelerated embeddings

Ollama running a local Mistral 7B language model

🖥️ Application Screenshots

Screenshots are stored in the screenshots/ folder.

Main UI
![alt text](<Screenshot (348).png>)

Sample query result
![alt text](<Screenshot (352).png>)

(These will render as images on GitHub if the paths and filenames match exactly.)

⚙️ Tech Stack

Python

FAISS

Sentence Transformers

PyTorch (GPU-enabled)

Streamlit

Ollama

Mistral 7B (local LLM)

✨ Features

GPU-accelerated text embeddings

Semantic search over SEC filings

Context-grounded answer generation

Bullet-point financial risk summaries

Fully offline execution

Modular and extensible RAG architecture

🚀 How to Run
1️⃣ Create environment

Create a Conda environment and install dependencies:

conda create -n rag_env python=3.10
conda activate rag_env
pip install -r requirements.txt

2️⃣ Install Ollama and pull the model

Install Ollama separately, then pull the model:

ollama pull mistral

3️⃣ Run the application

Start the Streamlit app:

streamlit run app.py

The application will be available at http://localhost:8501

📌 Example Query

What risks did Tesla disclose in 2023?

The system retrieves relevant sections from SEC filings and generates a grounded, bullet-point risk summary based strictly on retrieved context.

🧠 Future Improvements

Improved chunking strategy

Metadata-based filtering (company and year)

Hybrid search (BM25 + vector search)

Reranking retrieved chunks

Citation formatting per answer bullet

Answer confidence scoring

📄 Disclaimer

This project is for educational and research purposes only and does not constitute financial or investment advice.
