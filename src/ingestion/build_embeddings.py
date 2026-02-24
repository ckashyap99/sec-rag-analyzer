import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -------- Load chunks --------
with open("outputs\chunked_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]

# -------- Load embedding model --------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

print("Using device:", device)

# -------- Generate embeddings --------
embeddings = []

for text in tqdm(texts):
    emb = model.encode(
        text,
        normalize_embeddings=True
    )
    embeddings.append(emb)

embeddings = np.array(embeddings).astype("float32")

# -------- Build FAISS index --------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity (because normalized)
index.add(embeddings)

# -------- Save index --------
faiss.write_index(index, "outputs\sec_index.faiss")

# Save metadata
with open("outputs\sec_metadata.json", "w", encoding="utf-8") as f:
    json.dump(data, f)

print("Index built and saved successfully.")