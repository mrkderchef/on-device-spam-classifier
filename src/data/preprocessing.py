"""Text preprocessing pipeline for SMS spam detection."""

import re
import pandas as pd
from typing import List, Tuple


def clean_text(text: str) -> str:
    """Clean and normalize SMS text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)
    text = re.sub(r"\b\d{10,}\b", "[PHONE]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_sms_spam_dataset(filepath: str) -> pd.DataFrame:
    """Load the SMS Spam Collection dataset.

    Args:
        filepath: Path to the raw dataset file (TSV format).

    Returns:
        DataFrame with 'label' and 'text' columns.
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="latin-1",
    )
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def preprocess_dataset(
    df: pd.DataFrame, clean: bool = True
) -> pd.DataFrame:
    """Apply preprocessing to the dataset.

    Args:
        df: Raw dataset DataFrame.
        clean: Whether to apply text cleaning.

    Returns:
        Preprocessed DataFrame.
    """
    if clean:
        df["text"] = df["text"].apply(clean_text)
    return df


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets.

    Args:
        df: Full dataset DataFrame.
        test_size: Fraction for test set.
        val_size: Fraction for validation set.
        seed: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=train_df["label"],
    )
    return train_df, val_df, test_df
