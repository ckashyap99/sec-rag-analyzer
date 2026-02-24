#r"D:\conda_envs\RAG\DATA"
import os
import re
import json
from bs4 import BeautifulSoup

INPUT_DIR = r"D:\conda_envs\RAG\DATA"   # <-- your HTML folder
OUTPUT_FILE = "outputs\chunked_data.json"

# -----------------------------
# STEP 1 — Clean Text
# -----------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# STEP 2 — Detect ITEM Header
# -----------------------------
ITEM_PATTERN = re.compile(
    r'^item\s+(\d+[a-zA-Z]?)\.?\s*(.*)',
    re.IGNORECASE
)

def is_item_header(text):
    match = ITEM_PATTERN.match(text.lower())
    if match:
        item_number = match.group(1).upper()
        return item_number
    return None

# -----------------------------
# STEP 3 — Extract All ITEM Occurrences
# -----------------------------
def extract_largest_sections(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "lxml")

    body = soup.find("body")
    if not body:
        return {}

    # Flatten DOM into sequential text blocks
    elements = []
    for tag in body.find_all(True):
        text = tag.get_text(" ", strip=True)
        text = clean_text(text)
        if text:
            elements.append(text)

    # Find all ITEM header indices
    item_indices = []
    for i, text in enumerate(elements):
        item_number = is_item_header(text)
        if item_number:
            item_indices.append((item_number, i))

    # Add sentinel at end
    item_indices.append(("END", len(elements)))

    # Collect ALL sections
    sections_by_item = {}

    for idx in range(len(item_indices) - 1):
        item_number, start_idx = item_indices[idx]
        _, end_idx = item_indices[idx + 1]

        section_text = " ".join(elements[start_idx:end_idx])
        word_count = len(section_text.split())

        if item_number not in sections_by_item:
            sections_by_item[item_number] = []

        sections_by_item[item_number].append({
            "text": section_text,
            "word_count": word_count
        })

    # Keep ONLY largest section per item
    final_sections = {}
    for item_number, versions in sections_by_item.items():
        largest = max(versions, key=lambda x: x["word_count"])
        final_sections[item_number] = largest["text"]

    return final_sections

# -----------------------------
# STEP 4 — Chunking
# -----------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# -----------------------------
# STEP 5 — Process All Files
# -----------------------------
all_chunks = []

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".html"):
        path = os.path.join(INPUT_DIR, filename)
        print(f"\nProcessing: {filename}")

        sections = extract_largest_sections(path)

        for item, text in sections.items():
            word_count = len(text.split())
            print(f"Item {item} word count: {word_count}")

            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "company_file": filename,
                    "item": item,
                    "chunk_id": f"{filename}_ITEM{item}_{i}",
                    "text": chunk
                })

# -----------------------------
# STEP 6 — Save JSON
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"\nSaved {len(all_chunks)} chunks to {OUTPUT_FILE}")