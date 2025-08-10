import os, json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

from rag.utils import chunk_text

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR = Path("data/raw_docs")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BGE = "BAAI/bge-small-en-v1.5"

# rag/ingest/build_index.py
def load_text_from_pdf(path: Path) -> str:
    if path.suffix.lower() in {".md",".txt"}:
        return path.read_text(errors="ignore")
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""
# Reads .md and .txt directly as text.
# Reads .pdf using PyPDF2’s PdfReader, concatenating text from all pages.
# Returns the text string.
# Example usage:
# Page 1: "Machine learning is great."
# Page 2: "It powers chatbots like ChatGPT."

# "Machine learning is great.\nIt powers chatbots like ChatGPT."


def build_corpus() -> List[Dict]:
    docs = []
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}:
            text = load_text_from_pdf(p)
            if not text.strip():
                continue
            for i, chunk in enumerate(chunk_text(text)):
                docs.append({
                    "doc_path": str(p),
                    "chunk_id": f"{p.name}::chunk_{i}",
                    "text": chunk,
                })
    return docs

# Loops over every file in data/raw_docs (recursive).
# Reads it into text.
# Splits into chunks using chunk_text() (your earlier code with overlap).
# For each chunk, makes a dictionary:

# Example
# If sample.pdf text is "Word1 Word2 Word3 Word4 Word5 Word6"
# with chunk_size=4, overlap=2 → chunks would be:
# "Word1 Word2 Word3 Word4"
# "Word3 Word4 Word5 Word6"

# build_corpus() returns:
# [
#   {"doc_path": "data/raw_docs/sample.pdf", "chunk_id": "sample.pdf::chunk_0", "text": "Word1 Word2 Word3 Word4"},
#   {"doc_path": "data/raw_docs/sample.pdf", "chunk_id": "sample.pdf::chunk_1", "text": "Word3 Word4 Word5 Word6"}
# ]


def main():
    model_name = os.getenv("EMBED_MODEL", DEFAULT_BGE)
    print(f"Using embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Loads the embedding model (default "BAAI/bge-small-en-v1.5" unless overridden by EMBED_MODEL env var).

    corpus = build_corpus()
    if not corpus:
        raise SystemExit("No documents found to process. in data/raw_docs. Add PDFs/MD/TXT and rerun.")
    
    # Builds the chunked corpus from your raw docs.
    # Stops if no chunks found.
    # [
    #   {"doc_path": "file1.txt", "chunk_id": "chunk_0", "text": "AI is amazing"},
    #   {"doc_path": "file1.txt", "chunk_id": "chunk_1", "text": "It helps automation"}
    # ]

    
    texts = [c["text"] for c in corpus]
    # texts = ["AI is amazing", "It helps automation"]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    # [
    #     [0.12, -0.53, 0.77, ..., 0.05],   # "AI is amazing"
    #     [0.10, -0.50, 0.72, ..., 0.07]    # "It helps automation"
    # ]
    # If the model uses 384 dimensions, each list has 384 numbers.

    # Extracts just the text of each chunk.
    # Runs the model to convert them into embeddings (vector representations).
    # normalize_embeddings=True → makes them unit-length for cosine similarity.
    # Converts to NumPy array of type float32.  NumPy array (fast, efficient) and each number is float32 (small memory size).

    dim = embeddings.shape[1]

    # embeddings.shape tells you the size:
    # shape[0] = number of chunks
    # shape[1] = number of numbers per embedding (vector dimension)
    # Example: (2, 384) → 2 chunks, each vector length 384.

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # dim = number of dimensions in embedding vectors (e.g., 384 for MiniLM).
    # IndexFlatIP = FAISS index that uses inner product similarity (works like cosine similarity if vectors are normalized).
    # Adds all chunk embeddings to the index.  so you can later search them very quickly.

    faiss.write_index(index, str(OUT_DIR / "faiss_index.index"))
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    
    # Stores FAISS index to disk → data/processed/index.faiss
    # Saves meta.json so later you can map search results back to the original file + chunk.

    print(f"Index built with {len(corpus)} chunks, saved to {OUT_DIR / 'faiss_index.index'}")

if __name__ == "__main__":
    main()

# Example usage:
# Input
# data/raw_docs/
#   sample.txt   → "AI is amazing. It helps automate tasks."

# Output
# data/processed/
#   index.faiss   # Binary FAISS vector index
#   meta.json     # JSON mapping chunk IDs to text + source path

# meta.json
# [
#   {
#     "doc_path": "data/raw_docs/sample.txt",
#     "chunk_id": "sample.txt::chunk_0",
#     "text": "AI is amazing. It helps automate tasks."
#   }
# ]
