"""
Step 2: Clean the dataset — remove duplicates and handle data quality issues.

WHY we remove duplicates:
- 403 duplicate messages exist in the raw dataset.
- Duplicates can leak between train/test splits, causing overoptimistic evaluation.
- They artificially inflate the importance of certain patterns.
- For a fair distillation comparison, we need clean evaluation sets.

WHY we keep all messages (no outlier removal by length):
- Even very short messages ("Ok", "Yes") are realistic SMS data.
- Very long ham messages (up to 910 chars) exist but are genuine.
- Spam messages have a narrow length range (13-224 chars), no outliers.
- Removing messages would reduce an already small dataset.

DECISION: Remove exact text duplicates, keep first occurrence.
"""

import pandas as pd
from step_01_load_and_inspect import load_raw_dataset


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate text messages, keeping the first occurrence.
    
    Rationale: Duplicate messages could appear in both train and test sets,
    leading to data leakage and inflated performance metrics.
    
    Args:
        df: Raw DataFrame with potential duplicates.
        
    Returns:
        DataFrame with duplicates removed.
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    removed = initial_count - len(df_clean)
    print(f"  Removed {removed} duplicate messages ({initial_count} -> {len(df_clean)})")
    return df_clean


def verify_label_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Check if any duplicate texts had conflicting labels.
    
    If the same text appears with both 'ham' and 'spam' labels,
    that indicates a labeling error we should be aware of.
    
    Args:
        df: Raw DataFrame before deduplication.
        
    Returns:
        DataFrame of texts with conflicting labels (if any).
    """
    grouped = df.groupby("text")["label"].nunique()
    conflicting = grouped[grouped > 1]
    if len(conflicting) > 0:
        print(f"  WARNING: {len(conflicting)} texts have conflicting labels!")
        conflict_texts = df[df["text"].isin(conflicting.index)]
        return conflict_texts
    else:
        print("  No conflicting labels found — all duplicates have consistent labels.")
        return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)
    
    df = load_raw_dataset()
    
    print("\n[1] Checking for label conflicts in duplicates...")
    conflicts = verify_label_consistency(df)
    if len(conflicts) > 0:
        print(conflicts)
    
    print("\n[2] Removing duplicate messages...")
    df_clean = remove_duplicates(df)
    
    print(f"\n[3] Final dataset statistics:")
    print(f"  Total: {len(df_clean)}")
    print(f"  Ham:   {(df_clean['label'] == 'ham').sum()}")
    print(f"  Spam:  {(df_clean['label'] == 'spam').sum()}")
    print(f"  Spam ratio: {(df_clean['label'] == 'spam').mean():.4f}")
