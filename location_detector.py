import pandas as pd
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "shuffled_ward_duplication_data.csv"
df = pd.read_csv(DATA_PATH).dropna()

def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

df["variant_norm"] = df["variant"].apply(normalize)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(df["variant_norm"].tolist()).astype("float32")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

STOP_WORDS = {"hai","me","mein","ke","ki","ka","se","ko","aur","pani","problem"}

def extract_phrases(text):
    words = text.split()
    phrases = []
    for n in range(1, 4): # Greedy: Catch 1, 2, and 3 word phrases
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            if not any(sw in phrase.split() for sw in STOP_WORDS):
                phrases.append(phrase)
    return list(set(phrases))

def detect_location(text, threshold=0.85):
    text_norm = normalize(text)
    if not text_norm: return {"location_found": False}

    candidates = [text_norm] + extract_phrases(text_norm)
    best_score, best_row = 0, None

    for query in set(candidates):
        if not query.strip(): continue
        vec = model.encode([query]).astype("float32")
        faiss.normalize_L2(vec)
        scores, indices = index.search(vec, k=2)

        if scores[0][0] > threshold and scores[0][0] > best_score:
            # Check for identical confusion, otherwise accept best match
            if abs(scores[0][0] - scores[0][1]) < 0.0001: continue
            best_score = scores[0][0]
            best_row = df.iloc[indices[0][0]]

    if best_row is not None:
        return {
            "location_found": True,
            "ward_name": best_row["ward_name"],
            "ward_no": int(best_row["ward_no"]),
            "similarity_score": float(best_score)
        }
    return {"location_found": False}