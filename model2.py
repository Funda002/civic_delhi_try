import pandas as pd
import torch
import time
from tqdm import tqdm
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# --- SURGICAL FIX FOR THE MODEL CARD ERROR ---
from setfit.model_card import ModelCardCallback
ModelCardCallback.on_init_end = lambda *args, **kwargs: None
ModelCardCallback.on_train_finish = lambda *args, **kwargs: None
# ---------------------------------------------

def train_model():
    print("\n" + "="*50)
    print("STEP 1: LOADING AND PREPARING FULL DATA")
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
    
    # ⚠️ FEW-SHOT SAMPLING REMOVED! 
    # The model will now train on the FULL dataset.
    print(f"[OK] Training samples: {len(train_df)} | Test samples: {len(test_df)}")

    print("[*] Converting to Hugging Face Dataset format...")
    train_dataset = Dataset.from_pandas(train_df[['Text', 'label']].rename(columns={'Text': 'text'}))
    test_dataset = Dataset.from_pandas(test_df[['Text', 'label']].rename(columns={'Text': 'text'}))

    print("\n" + "="*50)
    print("STEP 2: LOADING TRANSFORMER MODEL TO GPU")
    print("="*50)
    
    print("[*] Checking GPU availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using Device: {device.upper()}")

    print("[*] Downloading/Loading 'paraphrase-multilingual-MiniLM-L12-v2'...")
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        labels=labels,
    )
    model.to(device) # Push model to GPU

    print("\n" + "="*50)
    print("STEP 3: INITIALIZING HIGH-PERFORMANCE TRAINER")
    print("="*50)
    
    args = TrainingArguments(
        batch_size=32,          # Doubled from 16 to utilize GPU VRAM
        num_iterations=20,      # Increased to SetFit's maximum default for highest accuracy
        num_epochs=10,          # Set to 10 Epochs as requested
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
    print("STEP 4: STARTING DEEP FINE-TUNING")
    print("="*50)
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print(f"\n[OK] Training complete in {round((end_time - start_time)/60, 2)} minutes.")

    print("\n" + "="*50)
    print("STEP 5: EVALUATING MODEL ON TEST DATA")
    print("="*50)
    
    test_texts = test_df['Text'].tolist()
    y_pred = []
    
    # Batch size increased to 64 for lightning-fast GPU inference
    inference_batch_size = 64 
    for i in tqdm(range(0, len(test_texts), inference_batch_size), desc="GPU Inference Progress"):
        batch = test_texts[i : i + inference_batch_size]
        preds = model.predict(batch)
        y_pred.extend(preds)

    y_true = test_df['Category'].tolist()
    
    acc = accuracy_score(y_true, y_pred)

    print(f"\n" + "!"*40)
    print(f"FINAL ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print("!"*40)

    print("\nDetailed Performance Report:")
    print(classification_report(y_true, y_pred))

    print("\n" + "="*50)
    print("STEP 6: SAVING MODEL ASSETS")
    print("="*50)
    save_path = "setfit_delhi_civic_model_10e"
    model.save_pretrained(save_path)
    print(f"[SUCCESS] High-Accuracy Model successfully saved to: ./{save_path}")
    print("="*50)

if __name__ == "__main__":
    train_model()