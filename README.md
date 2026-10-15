\# Local Qwen AI Chatbot



A local AI chatbot powered by \*\*Qwen2.5-0.5B-Instruct\*\*, Hugging Face Transformers, PyTorch, and Gradio.



This project demonstrates how to download an open-source language model and deploy it locally on a Windows PC without relying on a cloud inference API.



\## Features



\- Runs the Qwen2.5-0.5B-Instruct language model locally.

\- Provides a browser-based chat interface using Gradio.

\- Uses Hugging Face Transformers for model loading and text generation.

\- Supports CPU inference with PyTorch.

\- Includes a standalone script for testing model inference.



\## Architecture



User → Gradio Chat Interface → Python Application → Transformers → PyTorch → Qwen2.5-0.5B-Instruct



\## Technology Stack



\- Python

\- PyTorch

\- Hugging Face Transformers

\- Hugging Face Hub

\- Gradio



\## Requirements



\- Python 3.13

\- An internet connection for the initial model download

\- Sufficient RAM and disk space for the model and runtime



The current implementation uses CPU inference. Performance depends on your processor and available memory.



\## Installation



\### 1. Clone the repository



```bash

git clone https://github.com/YOUR-USERNAME/local-qwen-chatbot.git

cd local-qwen-chatbot

```



Replace `YOUR-USERNAME` with your GitHub username.



\### 2. Create a virtual environment



On Windows:



```bat

py -3.13 -m venv .venv

.venv\\Scripts\\activate

```



\### 3. Install dependencies



For the CPU-only PyTorch setup:



```bat

python -m pip install --upgrade pip

python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt

```



\### 4. Run the chatbot



```bat

python app.py

```



Open the local address printed by Gradio, usually:



http://127.0.0.1:7860



\### 5. Test model inference



```bat

python test\_qwen.py

```



The first run downloads the model from Hugging Face. Model files are cached locally and are not included in this Git repository.



\## Model



Qwen2.5-0.5B-Instruct



Model page: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct



\## Project Structure



```text

local-qwen-chatbot/

├── app.py

├── test\_qwen.py

├── requirements.txt

├── README.md

└── .gitignore

```



\## Future Improvements



\- Build a Retrieval-Augmented Generation (RAG) chatbot for PDF documents.

\- Add a vector database and semantic search.

\- Expose model inference through a REST API using FastAPI.

\- Optimize inference performance and memory usage.

\- Add evaluation and latency measurements.



\## Disclaimer



This is an educational project demonstrating local LLM inference. The model may produce inaccurate responses and should not be treated as a source of guaranteed factual information.



\## License



Review the model's license and the licenses of its dependencies before redistributing model files or using the model commercially.



