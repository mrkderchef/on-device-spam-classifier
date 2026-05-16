"""
Step 4: Encode labels — convert string labels to numeric format.

WHY numeric encoding:
- PyTorch and transformer models require numeric tensors.
- Binary classification: ham=0, spam=1
- This encoding is standard and interpretable.

WHY NOT one-hot encoding:
- For binary classification, a single integer label is sufficient.
- CrossEntropyLoss and KL-Divergence work with class indices.
- One-hot would add unnecessary complexity.

Label mapping:
  ham  → 0 (legitimate message, majority class)
  spam → 1 (spam/phishing message, minority class)
"""

import pandas as pd
from step_01_load_and_inspect import load_raw_dataset
from step_02_clean_duplicates import remove_duplicates
from step_03_text_normalization import normalize_text


LABEL_MAP = {"ham": 0, "spam": 1}
LABEL_MAP_INVERSE = {0: "ham", 1: "spam"}


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string labels to numeric format.
    
    Args:
        df: DataFrame with 'label' column containing 'ham'/'spam' strings.
        
    Returns:
        DataFrame with 'label' column as integers (0/1).
    """
    df = df.copy()
    df["label"] = df["label"].map(LABEL_MAP)
    
    # Validate: no unmapped labels
    assert df["label"].isnull().sum() == 0, "Found unmapped label values!"
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: LABEL ENCODING")
    print("=" * 60)
    
    df = load_raw_dataset()
    df = remove_duplicates(df)
    df["text"] = df["text"].apply(normalize_text)
    df = encode_labels(df)
    
    print(f"\nLabel distribution after encoding:")
    print(f"  0 (ham):  {(df['label'] == 0).sum()}")
    print(f"  1 (spam): {(df['label'] == 1).sum()}")
    print(f"\nSample:")
    print(df[["label", "text"]].head(10).to_string())
