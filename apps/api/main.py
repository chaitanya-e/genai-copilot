import os
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.retriever import Retriever

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

app = FastAPI(title="GenAI Copilot(Open Source)")
retriever = None

class AskReq(BaseModel):
    question: str
    k: int = 5

class AskResp(BaseModel):
    answer: str
    sources: List[Dict]
    latency_ms: float | None = None

# AskReq → client sends a question + optional k (how many relevant chunks to retrieve).
# AskResp → server responds with answer, sources, and latency.

@app.on_event("startup")
async def _startup():
    global retriever
    retriever = Retriever()
# When the app starts, it initializes your Retriever (likely loads embeddings/index).

@app.get("/healthz")
async def healthz():
    return {"ok": True}
# For monitoring — simply returns { "ok": true }.

async def ollama_generate(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
# Sends a POST request to Ollama’s API with your model and prompt.
# Returns the generated text.
    
SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant. Use ONLY the provided context to answer."
    "If the context contains any relevant information, summarize it clearly."
    "If nothing is directly related, summarize any relevant parts of the context."
    "Always cite sources in the format [doc_path::chunk_id]."
)

@app.post("/ask", response_model=AskResp)
async def ask(body: AskReq):
    from time import perf_counter
    start_time = perf_counter()

    hits = retriever.search(body.question, k=max(body.k, 5))
    # Starts timer for latency measurement.
    # Retrieves k most relevant document chunks from your retriever.

    context_blobs = []
    for h in hits:
        context_blobs.append(f"[SOURCE: {h['doc_path']} | {h['chunk_id']}]\n{h['text']}")
    context="\n\n".join(context_blobs)
    # Formats the retrieved chunks with citations like [SOURCE: file.pdf | chunk_3].

    prompt = f"""System: {SYSTEM_INSTRUCTIONS}

Context:
{context}

User question: {body.question}

Answer (with citations as [doc_path:: chunk_id]):
"""
# Creates the RAG prompt by:
# Adding system instructions.
# Adding the retrieved context.
# Adding the user’s question.
# Asking for citations.

    try:
        answer = await ollama_generate(prompt)
    except Exception as e:
        answer = (
            "(LLM Unavailable) Based on context snippets, here's what I found: \n\n" +
            "\n\n".join(cb[:600] for cb in context_blobs) 
        )
    # If Ollama is down, it falls back to returning just context snippets.

    dt = (perf_counter() - start_time) * 1000  # Convert to milliseconds

    uniq = {}
    for h in hits:
        uniq.setdefault(h['doc_path'], h)
    # Deduplicates sources by document path.

    sources = [
        {"doc_path": k, "chunk_id": v['chunk_id'], "score": v['score']}
        for k, v in uniq.items()
    ]        
    # Formats sources list for the response.
    return AskResp(
        answer=answer.strip(),
        sources=sources,
        latency_ms=dt
    )
    # Returns structured response.


# Example usage:
# meta.json
# [
#   {
#     "doc_path": "data/raw_docs/sample.txt",
#     "chunk_id": "sample.txt::chunk_0",
#     "text": "AI is amazing"
#   },
#   {
#     "doc_path": "data/raw_docs/sample.txt",
#     "chunk_id": "sample.txt::chunk_1",
#     "text": "It helps automate tasks"
#   },
#   {
#     "doc_path": "data/raw_docs/animals.txt",
#     "chunk_id": "animals.txt::chunk_0",
#     "text": "Cats love sleeping in the sun"
#   }
# ]

# POST /ask
# {
#   "question": "What is AI?",
#   "k": 2
# }

# hits = retriever.search(body.question, k=body.k)
# Here, retriever.search() finds the top-k chunks most relevant to "What is AI?".
# [
#   {
#     "doc_path": "data/raw_docs/sample.txt",
#     "chunk_id": "sample.txt::chunk_0",
#     "text": "AI is amazing",
#     "score": 0.95
#   },
#   {
#     "doc_path": "data/raw_docs/sample.txt",
#     "chunk_id": "sample.txt::chunk_1",
#     "text": "It helps automate tasks",
#     "score": 0.80
#   }
# ]

# The code collects these chunks into a context string:
# context_blobs = [
#     "[SOURCE: data/raw_docs/sample.txt | sample.txt::chunk_0]\nAI is amazing",
#     "[SOURCE: data/raw_docs/sample.txt | sample.txt::chunk_1]\nIt helps automate tasks"
# ]
# context = "\n\n".join(context_blobs)

# [SOURCE: data/raw_docs/sample.txt | sample.txt::chunk_0]
# AI is amazing

# [SOURCE: data/raw_docs/sample.txt | sample.txt::chunk_1]
# It helps automate tasks

# prompt = f"""System: You are a helpful assistant. Answer using ONLY the provided context...
# Context:
# {context}

# User question: What is AI?

# Answer (with citations as [doc_path:: chunk_id]):
# """
# Creates and Sends the prompt to your local Ollama server (qwen2.5:3b-instruct model).

# Say LLM Responds
# AI is amazing [data/raw_docs/sample.txt::sample.txt::chunk_0] and helps automate tasks [data/raw_docs/sample.txt::sample.txt::chunk_1].

# Prepare Unique Sources - The code keeps only one entry per doc_path:
# uniq = {
#   "data/raw_docs/sample.txt": {
#     "chunk_id": "sample.txt::chunk_0",
#     "score": 0.95
#   }
# }
# sources = [
#   {"doc_path": "data/raw_docs/sample.txt", "chunk_id": "sample.txt::chunk_0", "score": 0.95}
# ]

# Final response:
# {
#   "answer": "AI is amazing [data/raw_docs/sample.txt::sample.txt::chunk_0] and helps automate tasks [data/raw_docs/sample.txt::sample.txt::chunk_1].",
#   "sources": [
#     {
#       "doc_path": "data/raw_docs/sample.txt",
#       "chunk_id": "sample.txt::chunk_0",
#       "score": 0.95
#     }
#   ],
#   "latency_ms": 132.5
# }


