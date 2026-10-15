
from pathlib import Path
from pypdf import PdfReader

DOCUMENTS_DIR = Path("documents")
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30


def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n".join(pages)


def split_into_chunks(text):
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(words), step):
        chunk_words = words[start:start + CHUNK_SIZE]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

        if start + CHUNK_SIZE >= len(words):
            break

    return chunks


pdf_files = list(DOCUMENTS_DIR.rglob("*.pdf"))

if not pdf_files:
    print("No PDFs found.")
else:
    total_chunks = 0

    for pdf_file in pdf_files:
        text = extract_pdf_text(pdf_file)
        chunks = split_into_chunks(text)
        total_chunks += len(chunks)

        print(f"\nDocument: {pdf_file.name}")
        print(f"Total words: {len(text.split())}")
        print(f"Total chunks: {len(chunks)}")

        if chunks:
            print("\nFirst chunk preview:")
            print(chunks[0][:500])

    print(f"\nTotal chunks across all PDFs: {total_chunks}")
