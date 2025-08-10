import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1️⃣ Our "documents"
texts = [
    "AI is amazing",
    "It helps automate tasks",
    "Cats love sleeping in the sun"
]

# 2️⃣ Load a small embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3️⃣ Turn texts into embeddings
embeddings = model.encode(texts, normalize_embeddings=True)
embeddings = np.array(embeddings, dtype="float32")

print("Embeddings shape:", embeddings.shape)  # e.g., (3, 384)

# 4️⃣ Create FAISS index
dim = embeddings.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatIP(dim)  # IP = Inner Product
index.add(embeddings)  # Add our sentence vectors

# 5️⃣ Search with a query
query = "AI helps"
query_vector = model.encode([query], normalize_embeddings=True)
query_vector = np.array(query_vector, dtype="float32")

# Search: top 2 most similar sentences
scores, indices = index.search(query_vector, k=2)

print("Scores:", scores)
print("Indices:", indices)

# 6️⃣ Show the matching sentences
for idx, score in zip(indices[0], scores[0]):
    print(f"{texts[idx]} (score={score:.4f})")


# texts → embeddings
# The model converts sentences into 384-dimensional vectors.
# Similar meaning → closer vectors.
# Different meaning → farther apart.
# FAISS stores them so it can search really fast.
# Query "AI helps" also gets turned into an embedding.
# FAISS search finds which stored embeddings are most similar (highest cosine similarity score).

# Output:
# Embeddings shape: (3, 384)
# Scores: [[0.9041 0.7987]]
# Indices: [[1 0]]
# It helps automate tasks (score=0.9041)
# AI is amazing (score=0.7987)

# "It helps automate tasks" is the closest match to "AI helps".
# "AI is amazing" is the second closest.
# "Cats love sleeping in the sun" didn’t make the top 2 because it’s unrelated.