import json
from typing import List, Tuple, Dict, Any
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_BGE = "BAAI/bge-small-en-v1.5"
DATA_DIR = Path("data/processed")

class Retriever:
    def __init__(self, embed_model: str = DEFAULT_BGE):
        self.model = SentenceTransformer(embed_model)
        self.index = faiss.read_index(str(DATA_DIR / "faiss_index.index"))
        with open(DATA_DIR / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    # embed_model → Defaults to "BAAI/bge-small-en-v1.5" unless overridden.
    # self.model → Loads the sentence embedding model (same one used when building the index — otherwise results will be meaningless).
    # self.index → Reads the FAISS vector index from data/processed/faiss_index.index.
    # self.meta → Loads the list of metadata from meta.json (this is what maps embeddings back to text and original file path).

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(q, dtype="float32"), k)

        # query → The search string you type.
        # self.model.encode([query]) → Turns the query into an embedding vector (just like chunks during indexing).
        # normalize_embeddings=True → Makes similarity scores use cosine similarity.
        # self.index.search() → Searches the FAISS index:
        # D = similarity scores (higher = more relevant)
        # I = indices of the matching chunks in meta.json.

        hits = []
        for score, idx in zip(D[0], I[0]):
            # FAISS returns two arrays:
            # D = distances (or similarity scores) between your query and each retrieved vector.
            # I = indices (row numbers) of the vectors in your index that matched.
            # If you search with 1 query and ask for k=5 results:
            # D will have shape (1, 5) — 1 row, 5 scores.
            # I will have shape (1, 5) — 1 row, 5 indices.
            # D = [[0.95, 0.90, 0.85, 0.82, 0.80]] --> List of Lists. So we take the first row D[0].
            # I = [[12,   5,    19,   2,    8]] -> List of Lists. So we take the first row I[0].

            # zip(D[0], I[0])  
            # → (0.95, 12)  
            # → (0.90, 5)  
            # → (0.85, 19)  
            # → (0.82, 2)  
            # → (0.80, 8)


            rec = self.meta[int(idx)]
            hits.append({
                "score": float(score),
                "text": rec["text"],
                "doc_path": rec["doc_path"],
                "chunk_id": rec["chunk_id"]
            })
        return hits

        # Loops over each search result.
        # Looks up the chunk info in meta.json using idx.
        # Adds a dictionary with:
        # score → similarity score
        # text → actual chunk content
        # doc_path → original file path
        # chunk_id → chunk’s unique ID

# Example usage:
# meta.json
# [
#   {"doc_path": "data/raw_docs/sample.txt", "chunk_id": "sample.txt::chunk_0", "text": "AI is amazing"},
#   {"doc_path": "data/raw_docs/sample.txt", "chunk_id": "sample.txt::chunk_1", "text": "It helps automate tasks"},
#   {"doc_path": "data/raw_docs/animals.txt", "chunk_id": "animals.txt::chunk_0", "text": "Cats love sleeping in the sun"}
# ]

# faiss_index.index stores the embeddings for those 3 chunks.
    
# retriever = Retriever()
# results = retriever.search("AI helps", k=2)

# "AI helps" → embedding vector

# Compare to all embeddings in FAISS index

# Get top 2 matches:
# Scores: [0.9041, 0.7987]
# Indices: [1, 0]

# [
#   {
#     "score": 0.9041,
#     "text": "It helps automate tasks",
#     "doc_path": "data/raw_docs/sample.txt",
#     "chunk_id": "sample.txt::chunk_1"
#   },
#   {
#     "score": 0.7987,
#     "text": "AI is amazing",
#     "doc_path": "data/raw_docs/sample.txt",
#     "chunk_id": "sample.txt::chunk_0"
#   }
# ]
