
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Project paths
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "documents.faiss"
METADATA_PATH = BASE_DIR / "documents_metadata.npz"
CACHE_DIR = BASE_DIR / "hf_cache"

# Check that the RAG files exist
if not INDEX_PATH.exists() or not METADATA_PATH.exists():
    raise FileNotFoundError(
        "RAG index files are missing. Run test_faiss.py first."
    )

# Load the document search system
print("Loading embedding model...")
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=str(CACHE_DIR),
    device="cpu",
)

index = faiss.read_index(str(INDEX_PATH))
metadata = np.load(str(METADATA_PATH), allow_pickle=True)

print("Loading Qwen model...")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=str(CACHE_DIR),
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=str(CACHE_DIR),
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

model.eval()

print("RAG system loaded successfully!")


def retrieve_documents(question, top_k=3):
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal),
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "score": float(score),
            "text": str(metadata["chunks"][idx]),            
     "source": str(metadata["sources"][idx]),
        })

    return results


def ask_rag(question):
    passages = retrieve_documents(question)

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in passages
    )

    prompt = f"""You are a helpful assistant.
Answer the question using the supplied document context.
If the context does not contain the answer, say you could not find
the answer in the provided documents.
Do not invent facts.

Document context:
{context}

Question: {question}

Answer:"""

    messages = [
        {
            "role": "system",
            "content": "Answer using the provided document context."
        },
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    sources = "\n".join(
        f"- {item['source']} (similarity: {item['score']:.3f})"
        for item in passages
    )

    return f"{answer}\n\nSources used:\n{sources}"


if __name__ == "__main__":
    while True:
        question = input("\nAsk about your PDFs (or type exit): ").strip()

        if question.lower() == "exit":
            break

        if question:
            print("\nSearching documents and generating answer...\n")
            print(ask_rag(question))
