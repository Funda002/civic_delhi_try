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

def build_hierarchy(data_path="data/shuffled_ward_duplication_data.csv"):
    """
    Builds parent → child locality dictionary.
    Returns:
        parent_to_children: dict, e.g.,
            "lajpat nagar" -> ["lajpat nagar 1", "lajpat nagar 2", "amar colony"]
        locality_to_ward: dict, leaf locality -> ward info
    """

    # Load CSV
    df = pd.read_csv(data_path)

    # Clean column names: strip spaces, lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Adjust for your CSV columns
    required_cols = ['canonical_locality', 'ward_name', 'ward_no']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSV. Available columns: {df.columns.tolist()}")

    # Drop rows with missing data in required columns
    df = df.dropna(subset=required_cols)

    # Initialize dictionaries
    parent_to_children = defaultdict(list)
    locality_to_ward = {}

    # Clean localities
    df['locality_clean'] = df['canonical_locality'].apply(clean_text)

    for _, row in df.iterrows():
        locality = row['locality_clean']
        ward_name = row['ward_name']
        try:
            ward_no = int(row['ward_no'])
        except ValueError:
            ward_no = row['ward_no']

        # Map leaf locality → ward
        locality_to_ward[locality] = {
            "ward_name": ward_name,
            "ward_no": ward_no
        }

        # Derive parent locality by removing numbers, sector/block/phase/part
        parent = re.sub(r'\d+', '', locality)
        parent = re.sub(r'\b(sector|block|phase|part)\b', '', parent)
        parent = re.sub(r'\s+', ' ', parent).strip()

        if parent != locality:
            parent_to_children[parent].append(locality)

    # Deduplicate children
    parent_to_children = {k: sorted(list(set(v))) for k, v in parent_to_children.items()}

    return dict(parent_to_children), locality_to_ward

if __name__ == "__main__":
    parent_map, leaf_map = build_hierarchy()
    print("Sample parent → children (deduplicated):")
    for k, v in list(parent_map.items())[:10]:
        print(f"{k} -> {v}")