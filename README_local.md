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
rag.build_index -> utils.chunk_text()


corpus - collection of chunk texts
embeddings - collection of vectors for chunk texts
senetecetransformer - bulit on top of pytorch and huggingface to create embeddings
faiss - Facebook AI Similarity Search. It is used on top of embeddings. Its value can help in retrieval of nearest embeddings
