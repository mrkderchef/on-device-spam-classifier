"""
Step 7: Class imbalance analysis and strategy.

The SMS Spam Collection dataset is imbalanced:
  - Ham (legitimate): ~86.6% of messages
  - Spam: ~13.4% of messages

WHY this matters for our project:
- A naive model predicting "ham" always would achieve ~87% accuracy.
- Precision/Recall/F1 are more meaningful metrics than raw accuracy.
- The teacher model must learn to detect the minority class well.
- The student must retain this minority-class sensitivity through distillation.

STRATEGIES CONSIDERED:

1. Class-weighted loss (CHOSEN for teacher training):
   - Apply higher weight to the spam class in CrossEntropyLoss.
   - Weight = n_samples / (n_classes * n_class_samples)
   - This encourages the model to pay more attention to spam examples.
   - Simple, effective, no data augmentation needed.

2. Oversampling (NOT chosen):
   - Duplicate spam messages to balance classes.
   - Risk: overfitting to specific spam patterns.
   - With only 747 spam messages, repetition is problematic.

3. Undersampling (NOT chosen):
   - Remove ham messages to match spam count.
   - Would discard ~80% of the data — unacceptable with only 5K samples.

4. SMOTE / synthetic augmentation (NOT chosen):
   - Generating synthetic text is complex and unreliable.
   - Not standard practice for transformer-based NLP.
   - Could introduce artifacts.

5. No special handling (CHOSEN for distillation):
   - During distillation, the teacher's soft targets already encode
     the difficulty of spam detection.
   - The teacher naturally produces confident predictions on easy ham
     and more uncertain predictions near the decision boundary.
   - This "dark knowledge" helps the student without explicit weighting.

DECISION:
- Teacher training: Use class-weighted CrossEntropyLoss
- Student baseline: Use class-weighted CrossEntropyLoss (fair comparison)
- Distillation: No explicit class weighting (soft targets handle this implicitly)
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from step_01_load_and_inspect import load_raw_dataset
from step_02_clean_duplicates import remove_duplicates
from step_03_text_normalization import normalize_text
from step_04_label_encoding import encode_labels


def compute_class_weights(labels: np.ndarray) -> dict:
    """Compute balanced class weights using sklearn.
    
    Formula: weight_i = n_samples / (n_classes * n_samples_i)
    
    Args:
        labels: Array of integer labels.
        
    Returns:
        Dictionary mapping class index to weight.
    """
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 7: CLASS IMBALANCE ANALYSIS")
    print("=" * 60)
    
    df = load_raw_dataset()
    df = remove_duplicates(df)
    df["text"] = df["text"].apply(normalize_text)
    df = encode_labels(df)
    
    labels = df["label"].values
    
    print(f"\nClass distribution:")
    print(f"  Ham  (0): {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)")
    print(f"  Spam (1): {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
    
    weights = compute_class_weights(labels)
    print(f"\nComputed class weights (balanced):")
    print(f"  Ham  weight: {weights[0]:.4f}")
    print(f"  Spam weight: {weights[1]:.4f}")
    print(f"  Ratio (spam/ham): {weights[1]/weights[0]:.2f}x")
    
    print(f"\nThese weights will be used in CrossEntropyLoss:")
    print(f"  loss = CrossEntropyLoss(weight=torch.tensor([{weights[0]:.4f}, {weights[1]:.4f}]))")
    
    print(f"\n--- Naive baseline ---")
    print(f"  Always predict ham: {(labels == 0).mean()*100:.1f}% accuracy")
    print(f"  Always predict spam: {(labels == 1).mean()*100:.1f}% accuracy")
    print(f"  → Accuracy alone is misleading. F1-score is our primary metric.")
