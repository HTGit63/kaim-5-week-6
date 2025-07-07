import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss  # If using FAISS
import os
import re
import numpy as np
import time
from tqdm import tqdm  # Progress bar

# Load the cleaned data (with a subset for testing)
def load_data(path="data/filtered_complaints.csv"):
    print("Loading data...")
    df = pd.read_csv(path)
    
    # For testing purposes, limit to a smaller subset
    df = df.head(1000)  # Only load first 1000 rows for testing
    print(f"Loaded {df.shape[0]} records.")
    return df

# Basic exploratory data analysis (EDA)
def basic_eda(df):
    print(f"Total records: {df.shape[0]}, Columns: {df.shape[1]}\n")
    print("Product counts:")
    print(df['Product'].value_counts(), "\n")
    df['narrative_length'] = df['Consumer complaint narrative'].fillna("").apply(lambda x: len(str(x).split()))
    print("Narrative length summary:")
    print(df['narrative_length'].describe(), "\n")
    missing = df['Consumer complaint narrative'].isna().sum()
    total = df.shape[0]
    print(f"Missing narratives: {missing} ({missing/total:.2%})\n")

# Filter and clean the data
def filter_and_clean(df):
    print("Filtering and cleaning data...")
    target_products = {
        "Credit card", "Personal loan", "Buy Now, Pay Later (BNPL)",
        "Savings account", "Money transfers"
    }
    df = df[df["Product"].isin(target_products)].copy()
    df["Consumer complaint narrative"] = df["Consumer complaint narrative"].fillna("").astype(str)
    df = df[df["Consumer complaint narrative"].str.strip() != ""]
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"(i am writing to file a complaint:?)", "", text)
        text = re.sub(r"[^a-z0-9\s\.\,\!\?]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    
    df["cleaned_narrative"] = df["Consumer complaint narrative"].apply(clean_text)
    print(f"After filtering and cleaning: {df.shape[0]} records remain.")
    return df

# Chunk the text into smaller pieces
def chunk_text(df, column="cleaned_narrative", chunk_size=512, chunk_overlap=50):
    print("Starting text chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    df["chunks"] = df[column].apply(lambda x: splitter.split_text(x))
    print(f"Chunked {df.shape[0]} rows.")
    return df

# Initialize the embedding model (with a smaller, faster model)
def get_embeddings(model_name="distilbert-base-nli-mean-tokens"):
    print("Loading model...")
    model = SentenceTransformer(model_name)
    print("Model loaded successfully!")
    return model

# Embed text chunks and store in FAISS (or ChromaDB)
def embed_and_store(df, model, vector_store_path="vector_store/faiss_index", batch_size=32):
    all_embeddings = []
    all_metadata = []
    
    # Start timer for performance tracking
    start_time = time.time()

    # Process chunks in batches
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Chunks"):
        chunks = row["chunks"]
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            with torch.no_grad():
                embeddings = model.encode(batch, show_progress_bar=True)  # Batching the embedding process
            all_embeddings.extend(embeddings)
            for chunk, embedding in zip(batch, embeddings):
                all_metadata.append({
                    "complaint_id": row["Complaint ID"],  # Assuming you have an ID column
                    "product": row["Product"],            # Keep track of product for filtering
                    "chunk": chunk
                })
    
    # Convert to numpy array for FAISS indexing
    all_embeddings = np.array(all_embeddings).astype("float32")
    
    # Initialize FAISS index
    index = faiss.IndexFlatL2(all_embeddings.shape[1])  # L2 distance (Euclidean)
    index.add(all_embeddings)

    # Save FAISS index to disk
    if not os.path.exists(vector_store_path):
        os.makedirs(vector_store_path)
    faiss.write_index(index, os.path.join(vector_store_path, "index.faiss"))

    # Optionally save metadata for retrieval
    pd.DataFrame(all_metadata).to_csv(os.path.join(vector_store_path, "metadata.csv"), index=False)

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nFinished embedding {len(all_embeddings)} chunks in {elapsed_time:.2f} seconds.")
    return index, all_metadata

# Main process
if __name__ == "__main__":
    # Load and clean data
    df = load_data()
    basic_eda(df)
    df_clean = filter_and_clean(df)
    df_clean.to_csv("data/filtered_complaints.csv", index=False)
    
    # Chunking and embedding
    df_clean = chunk_text(df_clean)
    model = get_embeddings()
    embed_and_store(df_clean, model)
