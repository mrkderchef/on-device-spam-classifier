"""Teacher model fine-tuning on SMS spam dataset."""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from src.data.dataset import SMSSpamDataset
from src.models.teacher_model import create_teacher_model
from src.evaluation.metrics import compute_classification_metrics
from src.utils.training_logs import append_jsonl, make_log_paths, write_json


def train_teacher(
    data_dir: str = "data/processed",
    output_dir: str = "outputs/checkpoints",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    seed: int = 42,
):
    """Fine-tune the teacher model on the SMS spam dataset.

    Uses the preprocessed train/val CSVs produced by the preprocessing pipeline.
    Applies class-weighted CrossEntropyLoss to handle class imbalance
    (87.4% ham / 12.6% spam).

    Args:
        data_dir: Directory containing train.csv and val.csv.
        output_dir: Directory to save model checkpoints.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        max_length: Maximum token sequence length.
        seed: Random seed.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_paths = make_log_paths("teacher")
    print(f"Device: {device}")
    append_jsonl(
        log_paths["events"],
        {
            "event": "start",
            "device": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "seed": seed,
        },
    )

    # Load preprocessed splits
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples")

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: ham={class_weights[0]:.4f}, spam={class_weights[1]:.4f}")

    # Tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = SMSSpamDataset(train_df, tokenizer, max_length)
    val_dataset = SMSSpamDataset(val_df, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = create_teacher_model()
    model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Teacher parameters: {trainable_params:,}")

    # Weighted loss function
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        metrics = compute_classification_metrics(all_labels, all_preds)
        epoch_record = {"epoch": epoch + 1, "loss": avg_loss, **metrics}
        history.append(epoch_record)
        append_jsonl(log_paths["events"], {"event": "epoch", **epoch_record})
        print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f} | Val F1: {metrics['f1']:.4f}")

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            best_epoch = epoch + 1
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(os.path.join(output_dir, "teacher_best"))
            tokenizer.save_pretrained(os.path.join(output_dir, "teacher_best"))
            append_jsonl(
                log_paths["events"],
                {
                    "event": "checkpoint_saved",
                    "checkpoint": os.path.join(output_dir, "teacher_best"),
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                },
            )
            print(f"  Saved best teacher model (F1: {best_val_f1:.4f})")

    write_json(
        log_paths["summary"],
        {
            "run": "teacher",
            "device": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "seed": seed,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "trainable_params": trainable_params,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "history": history,
        },
    )
    append_jsonl(
        log_paths["events"],
        {"event": "complete", "best_epoch": best_epoch, "best_val_f1": best_val_f1},
    )
    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
    print(f"Summary written to: {log_paths['summary']}")


if __name__ == "__main__":
    train_teacher()
