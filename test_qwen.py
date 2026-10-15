
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Downloading and loading Qwen model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
)

model.eval()
print("Model loaded successfully!")

messages = [
    {
        "role": "user",
        "content": "Explain generative AI in simple terms."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

print("Generating response...")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

print("\nQwen response:")
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
