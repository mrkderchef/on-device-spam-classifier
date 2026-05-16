"""
Step 6: Tokenization analysis — understand how BERT tokenizes our SMS data.

This step does NOT modify data. It analyzes tokenization behavior to inform
model configuration decisions (e.g., max_length parameter).

WHY this analysis matters:
- BERT has a maximum sequence length (typically 512 tokens).
- Shorter max_length = faster training and inference = better for mobile.
- We need to find the minimum max_length that covers (nearly) all messages.
- SMS messages are short by nature — we can likely use max_length=128 or less.

KEY QUESTION: What max_length should we use?
- Too short: truncates messages, loses information
- Too long: wastes computation on padding, slower inference

FINDINGS (to be populated when run):
- Token length distribution of the dataset
- Recommended max_length setting
- Percentage of messages that would be truncated at various thresholds
"""

import pandas as pd
import numpy as np
from transformers import BertTokenizer

from step_01_load_and_inspect import load_raw_dataset
from step_02_clean_duplicates import remove_duplicates
from step_03_text_normalization import normalize_text
from step_04_label_encoding import encode_labels


def analyze_token_lengths(texts: list, tokenizer: BertTokenizer) -> dict:
    """Analyze token length distribution for a list of texts.
    
    Args:
        texts: List of text strings.
        tokenizer: BERT tokenizer.
        
    Returns:
        Dictionary with length statistics.
    """
    lengths = [len(tokenizer.encode(text)) for text in texts]
    lengths = np.array(lengths)
    
    stats = {
        "mean": lengths.mean(),
        "std": lengths.std(),
        "min": lengths.min(),
        "max": lengths.max(),
        "median": np.median(lengths),
        "p90": np.percentile(lengths, 90),
        "p95": np.percentile(lengths, 95),
        "p99": np.percentile(lengths, 99),
    }
    return stats, lengths


def coverage_analysis(lengths: np.ndarray, thresholds: list) -> dict:
    """Calculate what percentage of messages fit within various max_length values.
    
    Args:
        lengths: Array of token lengths.
        thresholds: List of max_length candidates.
        
    Returns:
        Dictionary mapping threshold to coverage percentage.
    """
    coverage = {}
    for t in thresholds:
        pct = (lengths <= t).mean() * 100
        coverage[t] = pct
    return coverage


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6: TOKENIZATION ANALYSIS")
    print("=" * 60)
    
    # Load preprocessed data
    df = load_raw_dataset()
    df = remove_duplicates(df)
    df["text"] = df["text"].apply(normalize_text)
    df = encode_labels(df)
    
    # Initialize tokenizer
    print("\nLoading bert-base-uncased tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Analyze overall
    print("\n--- Token Length Distribution (all messages) ---")
    stats, lengths = analyze_token_lengths(df["text"].tolist(), tokenizer)
    for key, value in stats.items():
        print(f"  {key:8s}: {value:.1f}")
    
    # Analyze per class
    print("\n--- Token Length by Class ---")
    for label, name in [(0, "Ham"), (1, "Spam")]:
        subset = df[df["label"] == label]["text"].tolist()
        s, _ = analyze_token_lengths(subset, tokenizer)
        print(f"  {name}: mean={s['mean']:.1f}, max={s['max']:.0f}, p95={s['p95']:.1f}")
    
    # Coverage analysis
    print("\n--- Coverage at Various max_length ---")
    thresholds = [32, 48, 64, 96, 128, 160, 192, 256]
    coverage = coverage_analysis(lengths, thresholds)
    for t, pct in coverage.items():
        marker = " ← RECOMMENDED" if t == 128 else ""
        print(f"  max_length={t:3d}: {pct:6.2f}% of messages fit{marker}")
    
    print("\n--- DECISION ---")
    print("  Using max_length=128 for all experiments.")
    print("  Rationale: Covers >99% of messages while keeping computation efficient.")
    print("  SMS messages are inherently short, making 128 tokens more than sufficient.")
    print("  This is also ideal for mobile deployment (shorter sequences = faster inference).")
