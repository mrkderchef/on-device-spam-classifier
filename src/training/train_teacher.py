"""Teacher model fine-tuning on SMS spam dataset."""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.preprocessing import load_sms_spam_dataset, preprocess_dataset, split_dataset
from src.data.dataset import SMSSpamDataset
from src.models.teacher_model import create_teacher_model
from src.evaluation.metrics import compute_classification_metrics


def train_teacher(
    data_path: str,
    output_dir: str = "outputs/checkpoints",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    seed: int = 42,
):
    """Fine-tune the teacher model on the SMS spam dataset.

    Args:
        data_path: Path to the raw SMS dataset.
        output_dir: Directory to save model checkpoints.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        max_length: Maximum token sequence length.
        seed: Random seed.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    df = load_sms_spam_dataset(data_path)
    df = preprocess_dataset(df)
    train_df, val_df, test_df = split_dataset(df, seed=seed)

    # Tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = SMSSpamDataset(train_df, tokenizer, max_length)
    val_dataset = SMSSpamDataset(val_df, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = create_teacher_model()
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

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
        print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f} | Val F1: {metrics['f1']:.4f}")

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(os.path.join(output_dir, "teacher_best"))
            tokenizer.save_pretrained(os.path.join(output_dir, "teacher_best"))
            print(f"  Saved best teacher model (F1: {best_val_f1:.4f})")

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train_teacher(data_path="data/raw/SMSSpamCollection")
