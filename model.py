import pandas as pd
import torch
import sys
import time
from tqdm import tqdm
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# --- SURGICAL FIX FOR THE MODEL CARD ERROR ---
from setfit.model_card import ModelCardCallback
# Redefine the buggy functions to do nothing
ModelCardCallback.on_init_end = lambda *args, **kwargs: None
ModelCardCallback.on_train_finish = lambda *args, **kwargs: None
# ---------------------------------------------

def train_model():
    print("\n" + "="*50)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*50)
    
    file_name = 'shuffled_final_seefit_data_5c_add1.csv'
    print(f"[*] Reading {file_name}...")
    df = pd.read_csv(file_name).dropna(subset=['Text', 'Category'])

    print("[*] Filtering rare categories...")
    counts = df['Category'].value_counts()
    df = df[~df['Category'].isin(counts[counts < 2].index)]

    print("[*] Encoding category labels...")
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Category'])
    labels = list(le.classes_)
    print(f"[OK] Found {len(labels)} unique categories.")

    print("[*] Splitting data into Train (80%) and Test (20%)...")
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    print(f"[OK] Training samples: {len(train_df)} | Test samples: {len(test_df)}")

    print("[*] Converting to Hugging Face Dataset format...")
    train_dataset = Dataset.from_pandas(train_df[['Text', 'label']].rename(columns={'Text': 'text'}))
    test_dataset = Dataset.from_pandas(test_df[['Text', 'label']].rename(columns={'Text': 'text'}))

    print("\n" + "="*50)
    print("STEP 2: LOADING TRANSFORMER MODEL")
    print("="*50)
    print("[*] Downloading/Loading 'paraphrase-multilingual-MiniLM-L12-v2'...")
    # This might take a moment depending on your internet
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        labels=labels,
    )

    print("\n" + "="*50)
    print("STEP 3: INITIALIZING TRAINER")
    print("="*50)
    args = TrainingArguments(
        batch_size=16,
        num_iterations=2, # Creates 2 pairs for every sample (~26,000 pairs total)
        num_epochs=1,
        body_learning_rate=2e-5,
        head_learning_rate=2e-3,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric="accuracy"
    )

    print("\n" + "="*50)
    print("STEP 4: STARTING FINE-TUNING (The Main Task)")
    print("="*50)
    print("[!] IMPORTANT: You will see two progress bars below.")
    print("[!] 1. 'Generating Pairs': Model is creating sentence matches.")
    print("[!] 2. 'Epochs': Model is learning from those matches.")
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print(f"\n[OK] Training complete in {round((end_time - start_time)/60, 2)} minutes.")

    print("\n" + "="*50)
    print("STEP 5: EVALUATING MODEL ON TEST DATA")
    print("="*50)
    print("[*] Running inference on test set (this shows the model's 'Guessing' speed)...")
    
    # We use a loop for predictions to show a manual progress bar
    test_texts = test_df['Text'].tolist()
    y_pred = []
    
    for i in tqdm(range(0, len(test_texts), 16), desc="Inference Progress"):
        batch = test_texts[i : i + 16]
        preds = model.predict(batch)
        y_pred.extend(preds)

    y_true = test_df['label'].tolist()
    acc = accuracy_score(y_true, y_pred)

    print(f"\n" + "!"*40)
    print(f"FINAL ACCURACY: {acc:.4f}")
    print("!"*40)

    print("\nDetailed Performance Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    print("\n" + "="*50)
    print("STEP 6: SAVING MODEL ASSETS")
    print("="*50)
    save_path = "setfit_delhi_civic_model"
    model.save_pretrained(save_path)
    print(f"[SUCCESS] Model saved to: ./{save_path}")
    print("="*50)

if __name__ == "__main__":
    train_model()