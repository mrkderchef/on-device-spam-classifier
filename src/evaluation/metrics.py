"""Classification and efficiency metrics."""

import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary of metric name to value.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true: List[int], y_pred: List[int]) -> str:
    """Get full classification report string."""
    return classification_report(
        y_true, y_pred, target_names=["ham", "spam"]
    )


def compute_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB (FP32)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Measure average inference latency.

    Args:
        model: Model to benchmark.
        input_ids: Input token IDs.
        attention_mask: Attention mask.
        num_runs: Number of inference runs.

    Returns:
        Dictionary with mean and std latency in milliseconds.
    """
    model.eval()
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    return {
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
    }


def compute_compression_ratio(teacher: nn.Module, student: nn.Module) -> float:
    """Compute compression ratio (teacher params / student params)."""
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    return teacher_params / student_params
