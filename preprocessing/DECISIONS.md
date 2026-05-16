# Preprocessing Decisions & Rationale

This document summarizes all preprocessing decisions made for the
**Knowledge Distillation for On-Device Spam Detection** project.

Each decision is justified in the context of:
- Transformer-based models (BERT teacher, lightweight student)
- Knowledge distillation training paradigm
- Mobile/edge deployment target
- Small dataset constraints (~5,500 messages)

---

## 1. Duplicate Removal

**Decision:** Remove 403 duplicate text messages (keep first occurrence).

**Rationale:**
- Duplicates risk appearing in both training and test sets → data leakage
- Artificially inflates importance of duplicated patterns
- After removal: 5,169 unique messages remain
- No label conflicts found in duplicates (all have consistent ham/spam labels)

---

## 2. Text Normalization (Minimal)

**Decision:** Apply minimal normalization only. No heavy NLP preprocessing.

**What we normalize:**
| Pattern | Replacement | Reason |
|---------|-------------|--------|
| URLs (http://, www.) | `[URL]` | Specific URL irrelevant; presence matters |
| Phone numbers (7+ digits) | `[PHONE]` | Specific number irrelevant; presence matters |
| Multiple whitespace | Single space | Noise reduction |

**What we deliberately DO NOT do:**
| Technique | Why NOT |
|-----------|---------|
| Lowercasing | BERT-uncased handles this internally |
| Stopword removal | Destroys sentence structure needed by attention |
| Stemming/Lemmatization | Conflicts with WordPiece tokenizer vocabulary |
| Punctuation removal | Punctuation carries meaning (!!!, ???, typical in spam) |
| Number removal | Numbers indicate spam patterns (prize amounts, dates) |

**Key insight:** BERT was pre-trained on raw text. Heavy preprocessing removes
information that the model learned to use. For transformer-based pipelines,
minimal normalization outperforms aggressive preprocessing.

---

## 3. Label Encoding

**Decision:** Binary encoding — ham=0, spam=1.

**Rationale:**
- PyTorch CrossEntropyLoss expects integer class indices
- Binary classification — single integer sufficient (no one-hot needed)
- Convention: 0=negative/majority class, 1=positive/minority class

---

## 4. Train/Validation/Test Split

**Decision:** 70% train / 15% validation / 15% test, stratified, seed=42.

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | ~3,618 | Model training (teacher and student) |
| Val | ~775 | Hyperparameter tuning, early stopping, temperature selection |
| Test | ~776 | Final evaluation only (never seen during training) |

**Stratification:** Preserves ~13.4% spam ratio in all splits.

**Why validation is needed:**
- Temperature sweep for distillation (T=1,2,4,8) requires validation-based selection
- Early stopping prevents overfitting on small dataset
- Fair comparison: same stopping criterion for teacher, student baseline, and distilled student

---

## 5. Sequence Length (max_length=128)

**Decision:** Use max_length=128 tokens for all models.

**Analysis results:**
- Mean token length: ~25 tokens (ham), ~40 tokens (spam)
- 95th percentile: ~50 tokens
- 99th percentile: ~80 tokens
- Maximum: ~180 tokens (very rare)

**128 tokens covers >99% of messages** with minimal padding waste.

**Mobile deployment benefit:** Shorter sequences = faster inference = lower latency.
A max_length of 128 is 4x shorter than BERT's maximum (512), making it significantly
more practical for edge deployment.

---

## 6. Class Imbalance Strategy

**Decision:** Class-weighted loss for teacher/student baseline; no explicit weighting for distillation.

**Dataset imbalance:** 86.6% ham / 13.4% spam

**Computed weights (sklearn balanced):**
- Ham weight: ~0.577
- Spam weight: ~3.726
- Ratio: spam class gets ~6.5x the loss contribution

**Why NOT during distillation:**
- Teacher soft targets already encode class difficulty
- Confident predictions on easy ham + uncertain predictions near boundary = built-in weighting
- Adding explicit weights on top of soft targets could cause overcorrection

**Primary metric:** F1-score (not accuracy!)
- A model predicting "ham" always gets 87% accuracy but 0% spam recall
- F1 balances precision and recall on the minority class

---

## 7. No Data Augmentation

**Decision:** No synthetic data generation or augmentation.

**Rationale:**
- Augmenting text for transformers is non-trivial and can introduce artifacts
- The dataset is well-curated (UCI benchmark)
- Knowledge distillation itself acts as a form of regularization
- The teacher's soft targets provide richer supervision than hard labels alone
- Focus is on distillation effectiveness, not data augmentation techniques

---

## Summary Pipeline

```
Raw SMS Spam Collection (5,572 messages)
    │
    ├── Remove 403 duplicates → 5,169 messages
    │
    ├── Normalize URLs → [URL]
    ├── Normalize phones → [PHONE]
    ├── Normalize whitespace
    │
    ├── Encode labels: ham=0, spam=1
    │
    └── Stratified split (seed=42)
         ├── train.csv  (70%, ~3,618 samples)
         ├── val.csv    (15%, ~775 samples)
         └── test.csv   (15%, ~776 samples)
```

All processed data saved in `data/processed/`.
