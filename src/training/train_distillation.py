"""Knowledge distillation training pipeline."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.preprocessing import load_sms_spam_dataset, preprocess_dataset, split_dataset
from src.data.dataset import SMSSpamDataset
from src.models.student_model import create_student_model, count_parameters
from src.evaluation.metrics import compute_classification_metrics


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute the combined distillation loss.

    L = alpha * T^2 * KL(softened_teacher || softened_student) + (1-alpha) * CE(student, labels)

    Args:
        student_logits: Raw logits from the student model.
        teacher_logits: Raw logits from the teacher model.
        labels: Ground truth labels.
        temperature: Softmax temperature for softening distributions.
        alpha: Weight balancing KL divergence and cross-entropy.

    Returns:
        Combined distillation loss.
    """
    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    kl_loss = F.kl_div(
        soft_student,
        soft_teacher,
        log_target=True,
        reduction="batchmean",
    )

    ce_loss = F.cross_entropy(student_logits, labels)

    loss = alpha * (temperature ** 2) * kl_loss + (1 - alpha) * ce_loss
    return loss


def train_distillation(
    data_path: str,
    teacher_path: str = "outputs/checkpoints/teacher_best",
    output_dir: str = "outputs/checkpoints",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    temperature: float = 4.0,
    alpha: float = 0.5,
    max_length: int = 128,
    seed: int = 42,
):
    """Train the student model using knowledge distillation from the teacher.

    Args:
        data_path: Path to the raw SMS dataset.
        teacher_path: Path to the fine-tuned teacher model.
        output_dir: Directory to save model checkpoints.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        temperature: Distillation temperature.
        alpha: Weight for KL divergence vs cross-entropy.
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
    tokenizer = BertTokenizer.from_pretrained(teacher_path)
    train_dataset = SMSSpamDataset(train_df, tokenizer, max_length)
    val_dataset = SMSSpamDataset(val_df, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Load teacher model (frozen)
    teacher = BertForSequenceClassification.from_pretrained(teacher_path)
    teacher.to(device)
    teacher.eval()

    # Create student model
    student = create_student_model(max_position_embeddings=max_length)
    student.to(device)
    print(f"Student model parameters: {count_parameters(student):,}")

    # Optimizer and scheduler
    optimizer = AdamW(student.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = 0.0
    for epoch in range(epochs):
        student.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Teacher forward (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            # Student forward
            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            # Distillation loss
            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                temperature=temperature, alpha=alpha,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        metrics = compute_classification_metrics(all_labels, all_preds)
        print(
            f"Epoch {epoch+1} — Loss: {avg_loss:.4f} | Val F1: {metrics['f1']:.4f} "
            f"(T={temperature}, alpha={alpha})"
        )

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "student_distilled_best")
            student.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  Saved best distilled student (F1: {best_val_f1:.4f})")

    print(f"\nDistillation complete. Best Val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train_distillation(data_path="data/raw/SMSSpamCollection")
