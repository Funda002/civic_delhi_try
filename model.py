import pandas as pd
import torch
import time
import os
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

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def train_model():
    total_start = time.time()

    print("\n" + "="*50)
    print("STEP 1: LOAD DATA")
    print("="*50)

    step_start = time.time()
    csv_path = "added_data.csv"
    log(f"Reading {csv_path} ...")
    df_raw = pd.read_csv(csv_path)
    log(f"Raw rows: {len(df_raw)} | Columns: {list(df_raw.columns)}")

    df = df_raw.dropna(subset=['text', 'category'])
    log(f"After dropna: {len(df)} (dropped {len(df_raw) - len(df)} rows with missing text/category)")

    counts = df['category'].value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        log(f"Removing {len(rare)} categories with < 2 samples: {rare}")
    df = df[~df['category'].isin(rare)]

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])
    labels = list(le.classes_)

    log(f"Final dataset: {len(df)} rows, {len(labels)} categories")
    log(f"Category distribution:\n{df['category'].value_counts().to_string()}")

    # TRAIN / VAL / TEST SPLIT
    train_df = df.sample(frac=0.7, random_state=42)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)

    log(f"Split -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    log(f"Train label distribution:\n{train_df['category'].value_counts().to_string()}")

    train_dataset = Dataset.from_pandas(train_df[['text','label']])
    val_dataset = Dataset.from_pandas(val_df[['text','label']])
    test_dataset = Dataset.from_pandas(test_df[['text','label']])

    log(f"Step 1 done in {time.time()-step_start:.2f}s")

    print("\n" + "="*50)
    print("STEP 2: LOAD MODEL")
    print("="*50)

    step_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    log(f"Loading pretrained model: {model_name}")
    model = SetFitModel.from_pretrained(model_name, labels=labels)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
    log(f"Step 2 done in {time.time()-step_start:.2f}s")

    print("\n" + "="*50)
    print("STEP 3: TRAINING SETUP")
    print("="*50)

    step_start = time.time()
    args = TrainingArguments(
        batch_size=32,
        num_iterations=20,
        num_epochs=3,
        body_learning_rate=2e-5,
        head_learning_rate=2e-3,
        seed=42,
    )

    log(f"batch_size={args.batch_size}, num_iterations={args.num_iterations}, "
        f"num_epochs={args.num_epochs}, body_lr={args.body_learning_rate}, "
        f"head_lr={args.head_learning_rate}, seed={args.seed}")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        metric="accuracy"
    )

    log(f"Trainer ready. Metric: accuracy")
    log(f"Step 3 done in {time.time()-step_start:.2f}s")

    print("\n" + "="*50)
    print("STEP 4: TRAIN")
    print("="*50)

    step_start = time.time()
    log("Training started...")
    if device == "cuda":
        log(f"GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, "
            f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

    trainer.train()

    train_elapsed = time.time() - step_start
    log(f"Training done in {train_elapsed/60:.2f} min ({train_elapsed:.1f}s)")
    if device == "cuda":
        log(f"GPU memory after training: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, "
            f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

    # Validation eval
    log("Running validation evaluation...")
    val_metrics = trainer.evaluate()
    log(f"Validation metrics: {val_metrics}")

    print("\n" + "="*50)
    print("STEP 5: FINAL TEST EVALUATION")
    print("="*50)

    step_start = time.time()
    test_texts = test_df['text'].tolist()
    y_true = test_df['category'].tolist()
    log(f"Predicting on {len(test_texts)} test samples...")

    probs = model.predict_proba(test_texts)
    y_pred = [labels[np.argmax(p)] for p in probs]

    pred_elapsed = time.time() - step_start
    log(f"Prediction done in {pred_elapsed:.2f}s ({len(test_texts)/pred_elapsed:.1f} samples/sec)")

    acc = accuracy_score(y_true, y_pred)
    log(f"FINAL TEST ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_true, y_pred))

    # Per-class accuracy
    log("Per-class accuracy:")
    for label in labels:
        mask = [t == label for t in y_true]
        if sum(mask) > 0:
            class_preds = [p for p, m in zip(y_pred, mask) if m]
            class_true = [t for t, m in zip(y_true, mask) if m]
            class_acc = accuracy_score(class_true, class_preds)
            log(f"  {label}: {class_acc:.4f} ({sum(mask)} samples)")

    # CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(xticks_rotation=90)
    plt.tight_layout()
    plt.savefig("confusion2.png", dpi=150)
    log("confusion2.png saved")

    # TOP-5 CONFIDENCE
    print("\nTOP-5 Predictions (sample):")
    for i, text in enumerate(test_texts[:5]):
        top5 = np.argsort(probs[i])[::-1][:5]
        print(f"\nInput: {text[:100]}{'...' if len(text) > 100 else ''}")
        for idx in top5:
            print(f"  {labels[idx]} -> {probs[i][idx]:.4f}")

    log(f"Step 5 done in {time.time()-step_start:.2f}s")

    print("\n" + "="*50)
    print("STEP 6: SAVE MODEL")
    print("="*50)

    step_start = time.time()
    save_path = "setfit_model_3e_val"
    log(f"Saving model to {save_path}/ ...")
    model.save_pretrained(save_path)

    model_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(save_path) for f in fns
    ) / 1e6
    log(f"Model saved: {model_size:.1f} MB")
    log(f"Step 6 done in {time.time()-step_start:.2f}s")

    total_elapsed = time.time() - total_start
    print("\n" + "="*50)
    log(f"ALL DONE in {total_elapsed/60:.2f} min ({total_elapsed:.1f}s)")
    print("="*50)

if __name__ == "__main__":
    train_model()
