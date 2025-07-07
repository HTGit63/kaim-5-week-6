# src/run_rag.py

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv

# 1. Load any .env (not strictly needed here, but kept for consistency)
load_dotenv()

# 2. Load FAISS index and metadata
def load_vector_store(path="vector_store/faiss_index"):
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    meta  = pd.read_csv(os.path.join(path, "metadata.csv"))
    return index, meta

# 3. Embed a query using the same model you used for indexing
def embed_query(query, model_name="distilbert-base-nli-mean-tokens"):
    model = SentenceTransformer(model_name)
    emb   = model.encode([query], show_progress_bar=False)[0].astype("float32")
    return emb

# 4. Retrieve top‑k chunks
def retrieve_chunks(query_emb, index, k=5):
    D, I = index.search(np.array([query_emb]), k)
    return I[0], D[0]

# 5. Format the retrieved chunks into a single context string
def build_context(meta, indices, distances):
    lines = []
    for idx, dist in zip(indices, distances):
        row = meta.iloc[idx]
        lines.append(
            f"- (Product: {row['product']}, ID: {row['complaint_id']}, dist={dist:.1f}): {row['chunk']}"
        )
    return "\n".join(lines)

# 6. Initialize a Hugging Face generation pipeline (FLAN‑T5‑small)
gen = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small",
    device=-1  # CPU only
)

# 7. Generate an answer using the retrieved context
def generate_answer(question, context,
                    system_prompt="You are a financial analyst assistant. Use only the provided excerpts to answer.",
                    max_length=128,
                    temperature=0.7):
    prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    outputs = gen(
        prompt,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        num_return_sequences=1
    )
    return outputs[0]["generated_text"].split("Answer:")[-1].strip()

# 8. Interactive loop
if __name__ == "__main__":
    idx, meta = load_vector_store()
    print("RAG system (DeepSeek→FLAN‑T5) ready. Type your question or ‘exit’.")
    while True:
        q = input("\n> ")
        if q.lower() in {"exit", "quit"}:
            break

        # embed, retrieve, build context
        q_emb    = embed_query(q)
        ids, ds  = retrieve_chunks(q_emb, idx, k=5)
        ctx      = build_context(meta, ids, ds)

        print("\n[Retrieved Context]")
        print(ctx, "\n")

        # generate and print answer
        ans = generate_answer(q, ctx)
        print(f"[Answer]\n{ans}\n")
