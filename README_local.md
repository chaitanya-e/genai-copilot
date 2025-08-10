# GenAI Copilot

A Retrieval Augmented Generation (RAG) system that allows you to query your documents using AI. The system combines document retrieval using FAISS with local LLM generation using Ollama.

## Features

- Document ingestion (PDF, TXT, MD files)
- Semantic search using FAISS and sentence transformers
- Local LLM integration with Ollama
- FastAPI-based REST API
- Chunk-based document processing with overlap
- Source attribution in responses

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- The Qwen 3B model pulled in Ollama (`ollama pull qwen2.5:3b-instruct`)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your documents in `data/raw_docs/` (supports PDF, TXT, MD)

2. Build the search index:
```bash
python -m rag.ingest.build_index
```

2a. Start the ollama LLM -- download from ollama.com
```bash
ollama pull qwen2.5:3b-instruct
ollama run qwen2.5:3b-instruct
```

3. Start the API server:
```bash
uvicorn apps.api.main:app --reload
```

4. Query your documents:
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" --data-raw '{"question": "Summarize scientific facts and artificial intelligence key points from my documents.", "k": 3}'
```
5. Docker commands:
```bash
docker build -f ops/docker/Dockerfile.api -t genai-api:local .
docker run --rm -p 8080:8080 -e OLLAMA_HOST=http://host.docker.internal:11434 genai-api:local
```

6. Execution with streamlit:
```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8080 
streamlit run apps/ui/app.py
```
7. MLOPS
```bash
export MLFLOW_TRACKING_URI=./mlruns --> Initialization of MLflow local
```
## API Endpoints

- `POST /ask`
  - Input: `{"question": "string", "k": int}`
  - Output: `{"answer": "string", "sources": [], "latency_ms": float}`
- `GET /healthz`
  - Health check endpoint

## Environment Variables

- `OLLAMA_HOST`: Ollama API host (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: Ollama model to use (default: `qwen2.5:3b-instruct`)
- `EMBED_MODEL`: Embedding model (default: `BAAI/bge-small-en-v1.5`)

## Project Structure

```
.
├── apps/
│   └── api/            # FastAPI application
├── rag/
│   ├── ingest/        # Document ingestion and indexing
│   ├── retriever.py   # FAISS-based document retrieval
│   └── utils.py       # Utility functions
├── data/
│   ├── raw_docs/      # Input documents
│   └── processed/     # FAISS index and metadata
└── eval/              # Evaluation scripts
```

## Behind the Scenes: API Request Flow

When you make a request like:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "k": 2}'
```

Here's what happens:

1. **FastAPI** (`apps/api/main.py::ask()`)
   - Receives POST request
   - Validates request body against AskReq model
   - Starts latency timer

2. **Retriever** (`rag/retriever.py::search()`)
   - Converts question to embedding vector using SentenceTransformer
   - Searches FAISS index for similar vectors
   - Returns top-k matching chunks with scores

3. **Context Building** (`apps/api/main.py`)
   - Formats retrieved chunks with source information
   - Builds prompt with system instructions and context

4. **LLM Generation** (`apps/api/main.py::ollama_generate()`)
   - Sends prompt to local Ollama server
   - Waits for generated response
   - Falls back to context snippets if LLM is unavailable

5. **Response Formation** (`apps/api/main.py::ask()`)
   - Deduplicates sources
   - Calculates total latency
   - Returns structured JSON response

### Example Flow with Data:

Input:
```json
{"question": "What is AI?", "k": 2}
```

Step 2 Output (Retriever):
```json
[
  {
    "doc_path": "data/raw_docs/ai.txt",
    "chunk_id": "ai.txt::chunk_0",
    "text": "AI is artificial intelligence",
    "score": 0.95
  },
  {
    "doc_path": "data/raw_docs/ai.txt",
    "chunk_id": "ai.txt::chunk_1",
    "text": "It simulates human thinking",
    "score": 0.85
  }
]
```

Step 3 Output (Context):
```text
[SOURCE: data/raw_docs/ai.txt | ai.txt::chunk_0]
AI is artificial intelligence

[SOURCE: data/raw_docs/ai.txt | ai.txt::chunk_1]
It simulates human thinking
```

Step 4 Output (LLM):
```text
Based on the provided context, AI (Artificial Intelligence) [data/raw_docs/ai.txt::ai.txt::chunk_0] 
is a technology that simulates human thinking [data/raw_docs/ai.txt::ai.txt::chunk_1].
```

Final Response:
```json
{
  "answer": "Based on the provided context, AI (Artificial Intelligence) [data/raw_docs/ai.txt::ai.txt::chunk_0] is a technology that simulates human thinking [data/raw_docs/ai.txt::ai.txt::chunk_1].",
  "sources": [
    {
      "doc_path": "data/raw_docs/ai.txt",
      "chunk_id": "ai.txt::chunk_0",
      "score": 0.95
    }
  ],
  "latency_ms": 145.2
}
```

## Flow chart
![alt text](image.png)

#### Program flow
```text
rag.build_index -> utils.chunk_text()
API CURL request -> main.get/post -> retriever

Glossary:
corpus - collection of chunk texts
embeddings - collection of vectors for chunk texts
sentencetransformer - bulit on top of pytorch and huggingface to create embeddings

PyTorch is like LEGO bricks for AI — you can snap together layers and functions to build whatever kind of AI model you want.
It’s an open-source machine learning framework (created by Facebook/Meta).
Developers and researchers use it to:
Build neural networks
Train models on data
Run those models to make predictions
It works on both CPU and GPU (GPUs make AI training way faster).
It’s written in Python (with C++ under the hood for speed).

If PyTorch is the toolbox, Hugging Face is the ready-made IKEA furniture store — you can just buy the model (for free) and use it, instead of crafting it from raw materials.
Hugging Face is like an AI model library + community hub.

PyTorch → the engine running the model (training & inference)
Hugging Face → the model provider & convenience tools for easy use
Example:
When you use a Hugging Face model, under the hood it often runs on PyTorch (or TensorFlow), so Hugging Face handles downloading and loading, and PyTorch does the math.


It’s a company & platform that hosts pretrained AI models (NLP, vision, speech, etc.).
It has a library called transformers that lets you download and use powerful models (like GPT, BERT, Stable Diffusion) with just a few lines of code.
You don’t have to train models from scratch — you can grab a ready-made one and fine-tune it for your data.

Streamlit is an open-source Python library that makes it super easy to build interactive web apps — especially for data science, machine learning, and visualization — without needing to know HTML, CSS, or JavaScript.

faiss - Facebook AI Similarity Search. It is used on top of embeddings. Its value can help in retrieval of nearest embeddings
```