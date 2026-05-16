"""
Step 5: Train/Validation/Test split — create proper evaluation sets.

SPLITTING STRATEGY:
- Train: 70% — used for model training
- Validation: 15% — used for hyperparameter tuning and early stopping
- Test: 15% — held out, used ONLY for final evaluation

WHY this specific split ratio:
- The dataset has only ~5,169 messages (after dedup).
- A 70/15/15 split gives ~3,618 train / ~775 val / ~776 test messages.
- This ensures enough test samples for reliable metric estimation.
- The validation set enables proper temperature tuning for distillation.

WHY stratified splitting:
- The dataset is imbalanced (only ~13.4% spam).
- Without stratification, some splits might have very few spam examples.
- Stratification preserves the class ratio in all splits.
- This ensures fair evaluation across all experimental conditions.

WHY a fixed random seed:
- Reproducibility: all experiments use the same splits.
- Fair comparison: teacher and student see the same training data.
- The seed (42) is arbitrary but fixed.

IMPORTANT: The test set must NEVER be used during training or hyperparameter
selection. It is only for final reported results.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from step_01_load_and_inspect import load_raw_dataset
from step_02_clean_duplicates import remove_duplicates
from step_03_text_normalization import normalize_text
from step_04_label_encoding import encode_labels


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple:
    """Create stratified train/validation/test splits.
    
    Args:
        df: Preprocessed DataFrame with numeric labels.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df["label"],
    )
    
    # Second split: separate validation from training
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_ratio,
        random_state=seed,
        stratify=train_val_df["label"],
    )
    
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                output_dir: str = "data/processed"):
    """Save splits to CSV files.
    
    Args:
        train_df: Training set.
        val_df: Validation set.
        test_df: Test set.
        output_dir: Directory to save CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"  Saved to {output_dir}/")
    print(f"    train.csv: {len(train_df)} samples")
    print(f"    val.csv:   {len(val_df)} samples")
    print(f"    test.csv:  {len(test_df)} samples")


def print_split_statistics(train_df, val_df, test_df):
    """Print detailed statistics for each split."""
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        total = len(split_df)
        spam = (split_df["label"] == 1).sum()
        ham = (split_df["label"] == 0).sum()
        print(f"  {name:5s}: {total:5d} samples | Ham: {ham:4d} ({ham/total*100:.1f}%) | Spam: {spam:3d} ({spam/total*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)
    
    # Load and preprocess
    df = load_raw_dataset()
    df = remove_duplicates(df)
    df["text"] = df["text"].apply(normalize_text)
    df = encode_labels(df)
    
    print(f"\nDataset after preprocessing: {len(df)} samples")
    print(f"Split ratios: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
    print(f"Random seed: {RANDOM_SEED}\n")
    
    # Create splits
    train_df, val_df, test_df = create_splits(df)
    
    print("Split statistics:")
    print_split_statistics(train_df, val_df, test_df)
    
    # Save
    print("\nSaving splits...")
    save_splits(train_df, val_df, test_df)
    
    # Verify no leakage
    train_texts = set(train_df["text"].values)
    val_texts = set(val_df["text"].values)
    test_texts = set(test_df["text"].values)
    
    assert len(train_texts & val_texts) == 0, "Data leakage: train/val overlap!"
    assert len(train_texts & test_texts) == 0, "Data leakage: train/test overlap!"
    assert len(val_texts & test_texts) == 0, "Data leakage: val/test overlap!"
    print("\n  No data leakage detected between splits.")
