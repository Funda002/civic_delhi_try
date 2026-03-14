# hierarchy_resolver.py

import pandas as pd
import re
from collections import defaultdict

def clean_text(text):
    """Lowercase, remove extra spaces and punctuation"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)     # collapse multiple spaces
    return text

def build_hierarchy(data_path="data/delhi_localities_gazetteer.csv"):
    """
    Builds parent → child locality dictionary from your gazetteer.
    """

    # Load CSV
    df = pd.read_csv(data_path)

    # Clean column names: strip spaces, lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Required columns based on your file
    required_cols = ['locality', 'ward_name', 'ward_no']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSV. Available columns: {df.columns.tolist()}")

    # Drop rows with missing data in required columns
    df = df.dropna(subset=required_cols)

    # Initialize dictionaries
    parent_to_children = defaultdict(list)
    locality_to_ward = {}

    # Clean localities
    df['locality_clean'] = df['locality'].apply(clean_text)

    for _, row in df.iterrows():
        locality = row['locality_clean']
        ward_name = row['ward_name']
        ward_no = row['ward_no']

        # Map leaf locality → ward
        locality_to_ward[locality] = {
            "ward_name": ward_name,
            "ward_no": ward_no
        }

        # Improved parent derivation: 
        # Removes common noise words to find the 'base' locality
        parent = re.sub(r'\d+', '', locality)
        parent = re.sub(r'\b(sector|block|phase|part|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z)\b', '', parent)
        parent = re.sub(r'\s+', ' ', parent).strip()

        # Only add to parent map if it's actually different from the base
        if parent and parent != locality:
            parent_to_children[parent].append(locality)

    # Deduplicate children and ensure sorting
    parent_to_children = {k: sorted(list(set(v))) for k, v in parent_to_children.items() if len(v) > 0}

    return dict(parent_to_children), locality_to_ward

if __name__ == "__main__":
    # Test with your specific file path
    parent_map, leaf_map = build_hierarchy("data/delhi_localities_gazetteer.csv")
    print(f"Loaded {len(leaf_map)} localities.")
    print("Sample parent → children:")
    for k, v in list(parent_map.items())[:5]:
        print(f"{k} -> {v}")