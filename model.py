import pandas as pd
import torch
import time
from tqdm import tqdm
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- SURGICAL FIX FOR THE MODEL CARD ERROR ---
from setfit.model_card import ModelCardCallback
ModelCardCallback.on_init_end = lambda *args, **kwargs: None
ModelCardCallback.on_train_finish = lambda *args, **kwargs: None
# ---------------------------------------------

def train_model():
    print("\n" + "="*50)
    print("STEP 1: LOADING AND PREPARING FULL DATA")
    print("="*50)
    
    file_name = 'added_data.csv'
    print(f"[*] Reading {file_name}...")
    df = pd.read_csv(file_name).dropna(subset=['text', 'category'])

    print("[*] Filtering rare categories...")
    counts = df['category'].value_counts()
    df = df[~df['category'].isin(counts[counts < 2].index)]

    print("[*] Encoding category labels...")
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])
    labels = list(le.classes_)
    print(f"[OK] Found {len(labels)} unique categories.")

    print("[*] Splitting data into Train (80%) and Test (20%)...")
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    print(f"[OK] Training samples: {len(train_df)} | Test samples: {len(test_df)}")

    print("[*] Converting to Hugging Face Dataset format...")
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    print("\n" + "="*50)
    print("STEP 2: LOADING TRANSFORMER MODEL TO GPU")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using Device: {device.upper()}")

    print("[*] Loading 'paraphrase-multilingual-MiniLM-L12-v2'...")
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        labels=labels,
    )
    model.to(device)

    print("\n" + "="*50)
    print("STEP 3: INITIALIZING TRAINER")
    print("="*50)
    
    args = TrainingArguments(
        batch_size=32,
        num_iterations=20,
        num_epochs=3,  # ✅ changed to 3 epochs
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
    print("STEP 4: STARTING TRAINING")
    print("="*50)
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print(f"\n[OK] Training complete in {round((end_time - start_time)/60, 2)} minutes.")

    print("\n" + "="*50)
    print("STEP 5: EVALUATING MODEL")
    print("="*50)
    
    test_texts = test_df['text'].tolist()
    y_true = test_df['category'].tolist()
    
    # Predict classes
    y_pred = model.predict(test_texts)

    # Accuracy & classification report
    acc = accuracy_score(y_true, y_pred)
    print(f"\nFINAL ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion2.png")
    print("[OK] Confusion matrix saved as confusion2.png")

    # Example: top-5 predictions with confidence
    print("\nSTEP 6: TOP-5 PREDICTIONS WITH CONFIDENCE SCORES (first 5 samples)")
    probs = model.model_body.predict_proba(test_texts)
    for i, text in enumerate(test_texts[:5]):
        top5_idx = np.argsort(probs[i])[::-1][:5]
        print(f"\nInput: {text}")
        for idx in top5_idx:
            print(f"  {labels[idx]} -> {probs[i][idx]:.4f}")

    print("\n" + "="*50)
    print("STEP 7: SAVING MODEL")
    save_path = "setfit_delhi_civic_model_3e"
    model.save_pretrained(save_path)
    print(f"[OK] Model saved to ./{save_path}")

if __name__ == "__main__":
    train_model()