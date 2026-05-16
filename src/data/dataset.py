"""PyTorch dataset classes for SMS spam detection."""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import pandas as pd
from typing import Dict


class SMSSpamDataset(Dataset):
    """PyTorch Dataset for SMS spam classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ):
        """
        Args:
            df: DataFrame with 'text' and 'label' columns.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum token sequence length.
        """
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
