
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading local Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32
)
model.eval()

print("Ready to chat!")


def respond(message, history):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Give clear, concise answers."
        }
    ]

    for item in history:
        if item["role"] in ("user", "assistant"):
            messages.append({
                "role": item["role"],
                "content": item["content"]
            })

    messages.append({
        "role": "user",
        "content": message
    })

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )



demo = gr.ChatInterface(
    fn=respond,
    title="My Local AI Assistant",
    description="Powered by Qwen2.5-0.5B-Instruct running on my PC"
)

if __name__ == "__main__":
    demo.launch()
