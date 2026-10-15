
from pathlib import Path
from pypdf import PdfReader

documents_folder = Path(
    r"C:\Users\pooji\qwen-local\documents"
)

# Search the folder and all its subfolders
pdf_files = list(documents_folder.rglob("*.pdf"))

if not pdf_files:
    print("No PDF files found.")
else:
    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        print(f"\nReading: {pdf_file.name}")

        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                print(f"\n--- Page {page_number} ---")
                print(text[:1000])

        print("\nPDF reading completed.")

