"""Lightweight student model for on-device spam classification."""

import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification


def create_student_model(
    num_labels: int = 2,
    num_layers: int = 3,
    hidden_size: int = 256,
    num_attention_heads: int = 4,
    intermediate_size: int = 512,
    vocab_size: int = 30522,
    max_position_embeddings: int = 128,
) -> BertForSequenceClassification:
    """Create a lightweight student model.

    Args:
        num_labels: Number of classification labels.
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size.
        num_attention_heads: Number of attention heads.
        intermediate_size: Feed-forward intermediate size.
        vocab_size: Vocabulary size (matches BERT tokenizer).
        max_position_embeddings: Maximum sequence length.

    Returns:
        Small BertForSequenceClassification model.
    """
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_labels=num_labels,
    )
    model = BertForSequenceClassification(config)
    return model


def get_student_config() -> dict:
    """Return student model configuration summary."""
    return {
        "num_layers": 3,
        "hidden_size": 256,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "estimated_parameters": "~5-15M",
        "estimated_fp32_size_mb": "~20-60",
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
