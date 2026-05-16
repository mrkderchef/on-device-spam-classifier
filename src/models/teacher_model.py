"""BERT-based teacher model for spam classification."""

import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig


def create_teacher_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
) -> BertForSequenceClassification:
    """Create the teacher model (BERT-base fine-tuned for classification).

    Args:
        model_name: Pretrained BERT model identifier.
        num_labels: Number of classification labels.

    Returns:
        BertForSequenceClassification model.
    """
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return model


def get_teacher_config() -> dict:
    """Return teacher model configuration summary."""
    return {
        "model_name": "bert-base-uncased",
        "num_parameters": 109_483_778,
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "fp32_size_mb": 417.6,
    }
