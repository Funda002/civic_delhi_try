import pandas as pd
import torch
import time
from tqdm import tqdm
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# --- FIX FOR MODEL CARD ERROR ---
from setfit.model_card import ModelCardCallback
ModelCardCallback.on_init_end = lambda *args, **kwargs: None
ModelCardCallback.on_train_finish = lambda *args, **kwargs: None
# --------------------------------

def train_model():
    print("\n" + "="*50)
    print("STEP 1: LOAD DATA")
    print("="*50)

    df = pd.read_csv("added_data.csv").dropna(subset=['text', 'category'])

    counts = df['category'].value_counts()
    df = df[~df['category'].isin(counts[counts < 2].index)]

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])
    labels = list(le.classes_)

    print(f"[OK] Found {len(labels)} categories")

    # ✅ TRAIN / VAL / TEST SPLIT
    train_df = df.sample(frac=0.7, random_state=42)
    temp_df = df.drop(train_df.index)

    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_dataset = Dataset.from_pandas(train_df[['text','label']])
    val_dataset = Dataset.from_pandas(val_df[['text','label']])
    test_dataset = Dataset.from_pandas(test_df[['text','label']])

    print("\n" + "="*50)
    print("STEP 2: LOAD MODEL")
    print("="*50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        labels=labels
    )
    model.to(device)

    print("\n" + "="*50)
    print("STEP 3: TRAINING SETUP")
    print("="*50)

    args = TrainingArguments(
        batch_size=32,
        num_iterations=20,
        num_epochs=3,
        body_learning_rate=2e-5,
        head_learning_rate=2e-3,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # ✅ VALIDATION HERE
        metric="accuracy"
    )

    print("\n" + "="*50)
    print("STEP 4: TRAIN")
    print("="*50)

    start = time.time()
    trainer.train()
    print(f"Training done in {(time.time()-start)/60:.2f} min")

    print("\n" + "="*50)
    print("STEP 5: FINAL TEST EVALUATION")
    print("="*50)

    test_texts = test_df['text'].tolist()
    y_true = test_df['category'].tolist()

    probs = model.predict_proba(test_texts)
    y_pred = [labels[np.argmax(p)] for p in probs]

    acc = accuracy_score(y_true, y_pred)
    print(f"\nFINAL TEST ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_true, y_pred))

    # ✅ CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(xticks_rotation=90)
    plt.tight_layout()
    plt.savefig("confusion2.png")
    print("[OK] confusion2.png saved")

    # ✅ TOP-5 CONFIDENCE
    print("\nTOP-5 Predictions (sample):")
    for i, text in enumerate(test_texts[:5]):
        top5 = np.argsort(probs[i])[::-1][:5]
        print(f"\nInput: {text}")
        for idx in top5:
            print(f"  {labels[idx]} -> {probs[i][idx]:.4f}")

    print("\n" + "="*50)
    print("STEP 6: SAVE MODEL")
    print("="*50)

    model.save_pretrained("setfit_model_3e_val")
    print("[OK] Model saved")

if __name__ == "__main__":
    train_model()