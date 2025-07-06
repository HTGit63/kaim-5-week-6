
import pandas as pd

def load_data(path="data/raw/complaints.csv"):
    df = pd.read_csv(path)
    return df

def basic_eda(df):
    # 1. Overall shape
    print(f"Total records: {df.shape[0]}, Columns: {df.shape[1]}\n")

    # 2. Product distribution
    print("Product counts:")
    print(df['Product'].value_counts(), "\n")

    # 3. Narrative length distribution
    #    (Assuming narratives in column 'Consumer complaint narrative')
    df['narrative_length'] = df['Consumer complaint narrative'].fillna("").apply(lambda x: len(str(x).split()))
    print("Narrative length summary:")
    print(df['narrative_length'].describe(), "\n")

    # 4. Missing narratives
    missing = df['Consumer complaint narrative'].isna().sum()
    total = df.shape[0]
    print(f"Missing narratives: {missing} ({missing/total:.2%})\n")

if __name__ == "__main__":
    df = load_data()
    basic_eda(df)

def filter_and_clean(df):
    # 1. Define the five products we care about
    target_products = {
        "Credit card",
        "Personal loan",
        "Buy Now, Pay Later (BNPL)",
        "Savings account",
        "Money transfers"
    }
    # 2. Keep only those products
    df = df[df["Product"].isin(target_products)].copy()
    
    # 3. Drop rows with missing or empty narratives
    df["Consumer complaint narrative"] = df["Consumer complaint narrative"].fillna("").astype(str)
    df = df[df["Consumer complaint narrative"].str.strip() != ""]
    
    # 4. Clean text: lowercase, remove boilerplate phrases and non‑alphanumeric chars
    import re
    def clean_text(text):
        text = text.lower()
        # remove common boilerplate
        text = re.sub(r"(i am writing to file a complaint:?)", "", text)
        # remove any character except letters, numbers, and basic punctuation
        text = re.sub(r"[^a-z0-9\s\.\,\!\?]", " ", text)
        # collapse whitespace
        return re.sub(r"\s+", " ", text).strip()
    
    df["cleaned_narrative"] = df["Consumer complaint narrative"].apply(clean_text)
    
    return df

if __name__ == "__main__":
    df = load_data()
    basic_eda(df)
    
    # Stage 3: filter & clean
    df_clean = filter_and_clean(df)
    print(f"After filtering: {df_clean.shape[0]} records remain.")
    
    # Save cleaned dataset
    df_clean.to_csv("data/filtered_complaints.csv", index=False)
    print("Saved filtered and cleaned data to data/filtered_complaints.csv")
