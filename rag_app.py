
from pathlib import Path

import os
os.environ["HF_HOME"] = str(Path(__file__).resolve().parent / "hf_cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"

import gradio as gr
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# 1. Project paths and model configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "hf_cache"

INDEX_PATH = BASE_DIR / "documents.faiss"
METADATA_PATH = BASE_DIR / "documents_metadata.npz"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

if not INDEX_PATH.exists() or not METADATA_PATH.exists():
    raise FileNotFoundError(
        "RAG index files are missing. Run test_faiss.py first."
    )

# --------------------------------------------------
# 2. Load the document search system
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_ID,
    cache_folder=str(CACHE_DIR),
    device="cpu",
)

index = faiss.read_index(str(INDEX_PATH))
metadata = np.load(METADATA_PATH, allow_pickle=True)

print("Loading Qwen model...")

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

print("Models and document index loaded successfully!")


# --------------------------------------------------
# 3. Retrieve relevant document passages
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Generate an answer using Qwen
# --------------------------------------------------

def answer_question(question):
    passages = retrieve_documents(question)

    if not passages:
        return "I couldn't find relevant passages in your documents."

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in passages
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer using the supplied "
                "document context. If the answer is not in the context, "
                "say you could not find it in the provided documents. "
                "Do not invent facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Document context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer the question based on the context."
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
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

    source_details = "\n\n".join(
        f"- **{item['source']}** "
        f"(similarity: {item['score']:.3f})\n"
        f"> {item['text'][:450]}"
        for item in passages
    )

    return (
        f"{answer}\n\n"
        f"---\n\n"
        f"**Retrieved sources**\n\n{source_details}"
    )


# --------------------------------------------------
# 5. Connect the chatbot to Gradio
# --------------------------------------------------

def chat(message, history):
    if not message.strip():
        return "Please enter a question."

    try:
        return answer_question(message.strip())
    except Exception as error:
        print(f"Error while answering: {error}")
        return f"Sorry, an error occurred: {error}"


demo = gr.ChatInterface(
    fn=chat,
    title="Local Qwen RAG Chatbot",
    description=(
        "Ask questions about your Python and Generative AI PDF library. "
        "Answers include retrieved source passages. "
        "Runs locally on your computer."
    ),
    examples=[
        "What is RAG and how does it work?",
        "Explain Python fundamentals.",
        "What should I study to become a GenAI engineer?",
    ],
)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
    )
