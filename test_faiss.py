
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CACHE_DIR = BASE_DIR / "hf_cache"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30

def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

def split_into_chunks(text):
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + CHUNK_SIZE])

        if chunk:
            chunks.append(chunk)

        if start + CHUNK_SIZE >= len(words):
            break

    return chunks

all_chunks = []
all_sources = []

for pdf_file in DOCUMENTS_DIR.rglob("*.pdf"):
    text = extract_pdf_text(pdf_file)

    for number, chunk in enumerate(split_into_chunks(text), start=1):
        all_chunks.append(chunk)
        all_sources.append(f"{pdf_file.name}, chunk {number}")

if not all_chunks:
    raise SystemExit("No document chunks found.")

print("Loading embedding model...")

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=str(CACHE_DIR),
    device="cpu"
)

print("Creating embeddings...")

embeddings = embedder.encode(
    all_chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

# Inner product on normalized vectors gives cosine similarity.
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save the index and its associated text and source metadata.
faiss.write_index(index, str(BASE_DIR / "documents.faiss"))
np.savez(
    BASE_DIR / "documents_metadata.npz",
    chunks=np.array(all_chunks),
    sources=np.array(all_sources)
)

print(f"FAISS index created with {index.ntotal} vectors.")
print("Saved documents.faiss and documents_metadata.npz")

# Test a semantic search.
question = "What should I learn to become a GenAI engineer?"

query_embedding = embedder.encode(
    [question],
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

scores, indices = index.search(query_embedding, k=min(3, len(all_chunks)))

print("\nTop matching passages:")

for score, idx in zip(scores[0], indices[0]):
    print(f"\nSimilarity: {score:.3f}")
    print(f"Source: {all_sources[idx]}")
    print(all_chunks[idx][:500])
