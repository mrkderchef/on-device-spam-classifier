"""
Step 8: Run the full preprocessing pipeline and produce final processed data.

This script chains all previous steps to create the final processed datasets
ready for model training. It serves as the single entry point to reproduce
the complete data preparation.

Pipeline:
  1. Load raw data (step_01)
  2. Remove duplicates (step_02)
  3. Normalize text (step_03)
  4. Encode labels (step_04)
  5. Create stratified splits (step_05)
  6. Save processed CSVs

Output files in data/processed/:
  - train.csv  (70% of data, stratified)
  - val.csv    (15% of data, stratified)
  - test.csv   (15% of data, stratified)
  
Each CSV has columns: label (int), text (str)
"""

import os
import sys
import pandas as pd

# Add preprocessing folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step_01_load_and_inspect import load_raw_dataset, inspect_dataset
from step_02_clean_duplicates import remove_duplicates, verify_label_consistency
from step_03_text_normalization import normalize_text
from step_04_label_encoding import encode_labels
from step_05_train_val_test_split import create_splits, save_splits, print_split_statistics


def run_full_pipeline(
    raw_data_path: str = None,
    output_dir: str = None,
):
    """Execute the complete preprocessing pipeline.
    
    Args:
        raw_data_path: Path to raw dataset.
        output_dir: Directory for processed output files.
    """
    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if raw_data_path is None:
        raw_data_path = os.path.join(project_root, "data", "raw", "SMSSpamCollection")
    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "processed")

    print("=" * 60)
    print("FULL PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load
    print("\n[Step 1] Loading raw dataset...")
    df = load_raw_dataset(raw_data_path)
    stats = inspect_dataset(df)
    print(f"  Loaded {stats['total_samples']} messages")
    
    # Step 2: Clean
    print("\n[Step 2] Removing duplicates...")
    verify_label_consistency(df)
    df = remove_duplicates(df)
    
    # Step 3: Normalize
    print("\n[Step 3] Normalizing text...")
    df["text"] = df["text"].apply(normalize_text)
    print(f"  Normalized {len(df)} messages")
    
    # Step 4: Encode
    print("\n[Step 4] Encoding labels...")
    df = encode_labels(df)
    print(f"  Labels: 0 (ham), 1 (spam)")
    
    # Step 5: Split
    print("\n[Step 5] Creating stratified splits (70/15/15)...")
    train_df, val_df, test_df = create_splits(df)
    print_split_statistics(train_df, val_df, test_df)
    
    # Save
    print("\n[Step 6] Saving processed data...")
    save_splits(train_df, val_df, test_df, output_dir)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_full_pipeline()
