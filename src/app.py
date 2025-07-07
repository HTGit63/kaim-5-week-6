# src/app.py

import os
import pandas as pd
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv

# Load environment (if needed)
load_dotenv()

# 1. Load FAISS index and metadata
def load_vector_store(path="vector_store/faiss_index"):
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    meta  = pd.read_csv(os.path.join(path, "metadata.csv"))
    return index, meta

# 2. Embed query using same model as indexing
def embed_query(query, model_name="distilbert-base-nli-mean-tokens"):
    embedder = SentenceTransformer(model_name)
    emb = embedder.encode([query], show_progress_bar=False)[0].astype("float32")
    return emb

# 3. Retrieve top-k chunks
def retrieve_chunks(query_emb, index, k=5):
    D, I = index.search(np.array([query_emb]), k)
    return I[0], D[0]

# 4. Format retrieved chunks
def build_context(meta, indices, distances):
    lines = []
    for idx, dist in zip(indices, distances):
        row = meta.iloc[idx]
        lines.append(f"- (Product: {row['product']}, ID: {row['complaint_id']}, dist={dist:.1f}): {row['chunk']}")
    return "\n".join(lines)

# 5. Initialize generation pipeline
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small",
    device=-1
)

# 6. Generate answer
def generate_answer(question, context,
                    system_prompt="You are a financial analyst assistant. Use only the provided excerpts to answer.",
                    max_new_tokens=50,
                    temperature=0.7):
    prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        num_return_sequences=1
    )
    return out[0]['generated_text'].split('Answer:')[-1].strip()

# Load once
di_index, di_meta = load_vector_store()
top_k = 5

# 7. Full RAG pipeline for UI
def rag_pipeline(question):
    q_emb = embed_query(question)
    ids, dists = retrieve_chunks(q_emb, di_index, top_k)
    ctx_text = build_context(di_meta, ids, dists)
    answer = generate_answer(question, ctx_text)
    return answer, ctx_text

# 8. Gradio interface
title = "CrediTrust RAG Chatbot"
desc  = "Ask questions about customer complaints across financial products."

demo = gr.Interface(
    fn=rag_pipeline,
    inputs=gr.Textbox(label="Your question"),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Retrieved Context")
    ],
    title=title,
    description=desc,
    theme="default",
)

if __name__ == "__main__":
    demo.launch()
