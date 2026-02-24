import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.llm.ollama_client import generate_answer

# ---------- Paths ----------
INDEX_PATH = "outputs/sec_index.faiss"
METADATA_PATH = "outputs/sec_metadata.json"

# ---------- Load index ----------
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ---------- Load embedding model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

print("Using device:", device)


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


# ---------- Query loop ----------
while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # Embed query
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    query_embedding = np.array([query_embedding]).astype("float32")

    # Search top 5
    distances, indices = index.search(query_embedding, 5)

    # Retrieve metadata
    top_results = [metadata[i] for i in indices[0]]
    # print("\nDEBUG SAMPLE RESULT:\n")
    # print(top_results[0])

    # Build prompt
    prompt = build_prompt(query, top_results)

    # Generate answer
    answer = generate_answer(prompt)

    print("\n=== AI Answer ===\n")
    print(answer)

    # print("\nTop Results:\n")

    # for rank, idx in enumerate(indices[0]):
    #     result = metadata[idx]
    #     print(f"Rank {rank+1}")
    #     print("File:", result["company_file"])
    #     print("Item:", result["item"])
    #     print("Text snippet:", result["text"][:400], "...\n")