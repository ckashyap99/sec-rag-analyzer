import streamlit as st
import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.llm.ollama_client import generate_answer

# ---------- Page Config ----------
st.set_page_config(page_title="SEC Risk Analyzer", layout="wide")

st.title("📊 SEC Risk Analysis Assistant")
st.write("Ask questions about company 10-K filings.")

# ---------- Load FAISS + Metadata ----------
@st.cache_resource
def load_resources():
    index = faiss.read_index("outputs/sec_index.faiss")
    with open("outputs/sec_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

    return index, metadata, model

index, metadata, model = load_resources()

# ---------- Prompt Builder ----------
def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(
        [
            f"[File: {chunk['company_file']} | Item: {chunk['item']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    return f"""
You are a financial analyst assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say you don't have enough information.

Context:
{context}

Question:
{question}

Provide:
- Bullet point summary
- Mention specific risk categories
- Avoid speculation
- Do not add information not present in the context
"""

# ---------- User Input ----------
query = st.text_input("Ask a question:")

if query:
    with st.spinner("Analyzing documents..."):

        query_embedding = model.encode(query, normalize_embeddings=True)
        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = index.search(query_embedding, 5)
        top_results = [metadata[i] for i in indices[0]]

        prompt = build_prompt(query, top_results)
        answer = generate_answer(prompt)

    st.subheader("AI Answer")
    st.write(answer)

    with st.expander("Retrieved Context"):
        for chunk in top_results:
            st.markdown(f"**{chunk['company_file']} | Item {chunk['item']}**")
            st.write(chunk["text"][:500] + "...")