"""
Step 1: Load and inspect the raw SMS Spam Collection dataset.

This script handles the initial data loading from the UCI-format TSV file
and provides a first look at dataset characteristics.

Key findings from inspection:
- 5,572 total messages (4,825 ham / 747 spam)
- Imbalanced dataset: only ~13.4% spam
- No null values
- 403 duplicate text messages exist
- Spam messages are typically longer (mean 139 chars) than ham (mean 72 chars)
- Text encoding: latin-1 (contains special characters like pound signs)
"""

import pandas as pd
import os

# Project root directory (one level up from preprocessing/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_raw_dataset(filepath: str = None) -> pd.DataFrame:
    """Load the raw SMS Spam Collection dataset.
    
    If no filepath is given, defaults to data/raw/SMSSpamCollection relative to project root.
    """
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, "data", "raw", "SMSSpamCollection")
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="latin-1",
    )
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Generate inspection statistics for the dataset.
    
    Returns:
        Dictionary with dataset statistics.
    """
    stats = {
        "total_samples": len(df),
        "ham_count": (df["label"] == "ham").sum(),
        "spam_count": (df["label"] == "spam").sum(),
        "spam_ratio": (df["label"] == "spam").mean(),
        "duplicates": df["text"].duplicated().sum(),
        "null_values": df.isnull().sum().sum(),
        "ham_avg_length": df[df["label"] == "ham"]["text"].str.len().mean(),
        "spam_avg_length": df[df["label"] == "spam"]["text"].str.len().mean(),
        "max_length": df["text"].str.len().max(),
        "min_length": df["text"].str.len().min(),
    }
    return stats


if __name__ == "__main__":
    df = load_raw_dataset()
    stats = inspect_dataset(df)
    
    print("=" * 60)
    print("SMS SPAM COLLECTION — RAW DATA INSPECTION")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n--- Sample ham messages ---")
    for text in df[df["label"] == "ham"]["text"].sample(3, random_state=42).values:
        print(f"  [{len(text):3d} chars] {text[:100]}")
    
    print("\n--- Sample spam messages ---")
    for text in df[df["label"] == "spam"]["text"].sample(3, random_state=42).values:
        print(f"  [{len(text):3d} chars] {text[:100]}")
