
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOCUMENTS_DIR = Path("documents")
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


# Collect chunks from every PDF, including subfolders.
all_chunks = []

for pdf_file in DOCUMENTS_DIR.rglob("*.pdf"):
    text = extract_pdf_text(pdf_file)
    all_chunks.extend(split_into_chunks(text))

if not all_chunks:
    print("No document chunks found.")
    raise SystemExit(1)

print(f"Total chunks: {len(all_chunks)}")
print("Loading local embedding model...")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "hf_cache"

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=str(CACHE_DIR),
    device="cpu"
)
print("Generating embeddings...")

embeddings = embedder.encode(
    all_chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embeddings generated successfully!")
print("Number of chunks:", len(all_chunks))
print("Embedding matrix shape:", embeddings.shape)
print("First embedding preview:", embeddings[0][:5])
