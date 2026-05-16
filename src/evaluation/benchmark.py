"""Model benchmarking utilities for comparing teacher and student models."""

import json
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from src.data.preprocessing import load_sms_spam_dataset, preprocess_dataset, split_dataset
from src.data.dataset import SMSSpamDataset
from src.models.student_model import create_student_model
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_model_size_mb,
    count_parameters,
    measure_inference_latency,
    compute_compression_ratio,
)


def benchmark_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str = "model",
) -> Dict:
    """Run full benchmark on a model.

    Args:
        model: Model to benchmark.
        dataloader: Test data loader.
        device: Device to run on.
        model_name: Name for logging.

    Returns:
        Dictionary with all benchmark results.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Classification metrics
    clf_metrics = compute_classification_metrics(all_labels, all_preds)

    # Efficiency metrics
    sample_batch = next(iter(dataloader))
    sample_input = sample_batch["input_ids"][:1].to(device)
    sample_mask = sample_batch["attention_mask"][:1].to(device)

    latency = measure_inference_latency(model, sample_input, sample_mask)

    results = {
        "model_name": model_name,
        "parameters": count_parameters(model),
        "model_size_mb": compute_model_size_mb(model),
        **clf_metrics,
        **latency,
    }

    print(f"\n{'='*50}")
    print(f"Benchmark: {model_name}")
    print(f"{'='*50}")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    return results


def run_full_comparison(
    data_path: str,
    teacher_path: str = "outputs/checkpoints/teacher_best",
    student_baseline_path: str = "outputs/checkpoints/student_baseline_best",
    student_distilled_path: str = "outputs/checkpoints/student_distilled_best",
    output_path: str = "outputs/logs/benchmark_results.json",
    max_length: int = 128,
    batch_size: int = 32,
    seed: int = 42,
):
    """Run benchmark comparison across all models.

    Args:
        data_path: Path to raw SMS dataset.
        teacher_path: Path to saved teacher model.
        student_baseline_path: Path to saved student baseline model.
        student_distilled_path: Path to saved distilled student model.
        output_path: Path to save benchmark results.
        max_length: Maximum sequence length.
        batch_size: Batch size for evaluation.
        seed: Random seed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    df = load_sms_spam_dataset(data_path)
    df = preprocess_dataset(df)
    _, _, test_df = split_dataset(df, seed=seed)

    tokenizer = BertTokenizer.from_pretrained(teacher_path)
    test_dataset = SMSSpamDataset(test_df, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    results = {}

    # Benchmark teacher
    teacher = BertForSequenceClassification.from_pretrained(teacher_path)
    teacher.to(device)
    results["teacher"] = benchmark_model(teacher, test_loader, device, "Teacher (BERT-base)")
    del teacher

    # Benchmark student baseline
    student_base = BertForSequenceClassification.from_pretrained(student_baseline_path)
    student_base.to(device)
    results["student_baseline"] = benchmark_model(
        student_base, test_loader, device, "Student Baseline"
    )
    del student_base

    # Benchmark distilled student
    student_kd = BertForSequenceClassification.from_pretrained(student_distilled_path)
    student_kd.to(device)
    results["student_distilled"] = benchmark_model(
        student_kd, test_loader, device, "Student Distilled"
    )
    del student_kd

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_full_comparison(data_path="data/raw/SMSSpamCollection")
