

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# 1. Load metadata and FAISS index
def load_vector_store(vector_store_path="vector_store/faiss_index"):
    # Load FAISS index
    index = faiss.read_index(os.path.join(vector_store_path, "index.faiss"))
    # Load metadata
    meta = pd.read_csv(os.path.join(vector_store_path, "metadata.csv"))
    return index, meta

# 2. Embed a user query
# src/retrieve.py

# 2. Embed a user query with the same model you used to index
def embed_query(query, model_name="distilbert-base-nli-mean-tokens"):
    model = SentenceTransformer(model_name)
    emb = model.encode([query], show_progress_bar=False)[0].astype("float32")
    return emb


# 3. Retrieve top‑k chunks
def retrieve_chunks(query_embedding, index, top_k=5):
    D, I = index.search(np.array([query_embedding]), top_k)
    return I[0], D[0]  # indices and distances

# 4. Display results
def display_results(indices, distances, metadata):
    for idx, dist in zip(indices, distances):
        row = metadata.iloc[idx]
        print(f"— Chunk (Product: {row['product']}, ID: {row['complaint_id']}, dist={dist:.4f}):")
        print(f"{row['chunk']}\n")

if __name__ == "__main__":
    # Example workflow
    index, metadata = load_vector_store()
    while True:
        q = input("\nEnter a question (or 'exit' to quit): ")
        if q.lower() in {"exit", "quit"}:
            break
        qe = embed_query(q)
        idxs, dists = retrieve_chunks(qe, index, top_k=5)
        display_results(idxs, dists, metadata)
