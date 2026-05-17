# Knowledge Distillation for On-Device Spam Detection

<p align="center">
  <strong>Compressing BERT into a mobile-friendly spam classifier with zero performance loss</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/F1_Score-0.9645-brightgreen" alt="F1">
  <img src="https://img.shields.io/badge/Compression-11.5x-blue" alt="Compression">
  <img src="https://img.shields.io/badge/Size_Reduction-91.3%25-orange" alt="Size">
  <img src="https://img.shields.io/badge/Python-3.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" alt="PyTorch">
</p>

---

## Overview

This project demonstrates that **knowledge distillation** can compress a full BERT-base model (109.5M parameters, 417.6 MB) into a lightweight Mini-BERT student (9.5M parameters, 36.2 MB) with **zero F1 score loss** for SMS spam detection — achieving a theoretical 11.5x compression suitable for on-device mobile deployment.

### Research Question

> Can a lightweight distilled transformer model retain strong spam/phishing detection performance while reducing model size by >90% for mobile deployment?

**Answer: Yes.** The distilled student achieves identical F1=0.9645 to the teacher while being 11.5x smaller.

---

## Results at a Glance

| Model | Parameters | Size | Test Accuracy | Test F1 | Precision | Recall |
|-------|-----------|------|---------------|---------|-----------|--------|
| **Teacher (BERT-base)** | 109,483,778 | 417.6 MB | 99.10% | 0.9645 | 0.9596 | 0.9694 |
| **Student Baseline** | 9,495,042 | 36.2 MB | 99.10% | 0.9637 | 0.9789 | 0.9490 |
| **Student Distilled** | 9,495,042 | 36.2 MB | 99.10% | **0.9645** | 0.9596 | 0.9694 |

**Key Finding:** The distilled student not only matches the teacher's accuracy — it replicates the teacher's exact confusion matrix pattern, demonstrating that knowledge distillation transfers decision *behavior*, not just accuracy.

---

## Motivation & Reasoning

### Why On-Device Spam Detection?

1. **Privacy**: SMS content never leaves the device — no cloud processing of personal messages
2. **Latency**: Real-time classification without network round-trips (critical for incoming message filtering)
3. **Offline capability**: Works in airplane mode, poor connectivity, or restricted networks
4. **Cost**: No per-inference cloud API costs for millions of messages
5. **Security**: No attack surface from network transmission of sensitive message content

### Why Knowledge Distillation?

Rather than training a small model from scratch (which often sacrifices quality) or deploying a full BERT model (impractical on mobile), distillation offers the best of both worlds:

- **Teacher quality** in a **student-sized package**
- Soft targets from the teacher provide richer supervision than hard labels alone
- The student learns the teacher's *confidence distribution*, not just binary decisions
- Temperature scaling (T=4.0) amplifies information in the teacher's probability tails

### Why BERT-base as Teacher?

- State-of-the-art for text classification at this scale
- Well-understood architecture with reproducible results
- Pre-trained representations capture deep linguistic patterns
- Sufficient capacity to learn near-perfect spam detection

### Why This Specific Student Architecture?

The Mini-BERT student (3 layers, 256 hidden, 4 heads) was designed with mobile constraints:

| Constraint | Teacher | Student | Reasoning |
|-----------|---------|---------|-----------|
| Layers | 12 | 3 | Diminishing returns beyond 3 for this task complexity |
| Hidden Size | 768 | 256 | SMS messages are short; 256 captures sufficient semantics |
| Attention Heads | 12 | 4 | Reduced heads still capture key attention patterns |
| Max Sequence | 512 | 128 | 99.8% of SMS fit in 128 tokens |
| Size | 417.6 MB | 36.2 MB | Fits comfortably in mobile memory constraints |

---

## Dataset

### SMS Spam Collection (UCI Machine Learning Repository)

| Property | Value |
|----------|-------|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) |
| Raw samples | 5,572 |
| After deduplication | 5,169 (403 duplicates removed) |
| Classes | Binary: Ham (87.4%) / Spam (12.6%) |
| Language | English |
| Average length | ~80 characters |

### Preprocessing Pipeline

A rigorous 8-step preprocessing pipeline ensures data quality and reproducibility:

| Step | Operation | Reasoning |
|------|-----------|-----------|
| 01 | Load raw data | Parse tab-separated format from UCI source |
| 02 | Deduplicate | Remove 403 exact duplicates to prevent train/test leakage |
| 03 | Normalize text | Standardize URLs → `[URL]`, phones → `[PHONE]`, collapse whitespace |
| 04 | Encode labels | ham=0, spam=1 for PyTorch compatibility |
| 05 | Stratified split | 70/15/15 train/val/test with preserved class ratios |
| 06 | Validate splits | Confirm no data leakage between splits |
| 07 | Save processed | CSV format for universal compatibility |
| 08 | Generate stats | Dataset summary for documentation |

**Split sizes:** Train=3,617 | Validation=776 | Test=776

All preprocessing decisions are documented in [`preprocessing/DECISIONS.md`](preprocessing/DECISIONS.md).

---

## Architecture Details

### Teacher Model: BERT-base-uncased

```
BertForSequenceClassification(
  bert: BertModel(
    embeddings: BertEmbeddings(vocab=30522, hidden=768, max_pos=512)
    encoder: 12× BertLayer(
      attention: MultiHeadAttention(12 heads, 768 dim)
      intermediate: Linear(768 → 3072, GELU)
      output: Linear(3072 → 768)
    )
    pooler: Linear(768 → 768, Tanh)
  )
  classifier: Linear(768 → 2)
)
Total Parameters: 109,483,778
```

### Student Model: Mini-BERT

```
MiniBertForClassification(
  bert: BertModel(
    embeddings: BertEmbeddings(vocab=30522, hidden=256, max_pos=128)
    encoder: 3× BertLayer(
      attention: MultiHeadAttention(4 heads, 256 dim)
      intermediate: Linear(256 → 1024, GELU)
      output: Linear(1024 → 256)
    )
    pooler: Linear(256 → 256, Tanh)
  )
  classifier: Sequential(
    Linear(256 → 128, ReLU)
    Dropout(0.1)
    Linear(128 → 2)
  )
)
Total Parameters: 9,495,042
```

---

## Training Methodology

### Teacher Fine-Tuning

- **Base model:** `bert-base-uncased` (HuggingFace)
- **Epochs:** 5
- **Learning rate:** 2e-5 (AdamW with linear warmup)
- **Batch size:** 32
- **Loss:** Cross-Entropy with class weights (ham=0.5723, spam=3.9579)
- **Rationale for class weights:** Severe class imbalance (7:1 ham:spam) would bias model toward majority class without compensation

### Student Baseline Training

- **Epochs:** 10 (best at epoch 6)
- **Learning rate:** 5e-4 (higher LR for smaller model)
- **Loss:** Cross-Entropy with class weights
- **Purpose:** Establish what the student architecture achieves without distillation knowledge

### Knowledge Distillation Training

- **Epochs:** 15 (best at epoch 3 — faster convergence!)
- **Learning rate:** 5e-4
- **Temperature (T):** 4.0
- **Alpha (α):** 0.5
- **Loss function:**

$$\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}\left(\frac{\text{softmax}(z_t/T)}{\text{softmax}(z_s/T)}\right) + (1-\alpha) \cdot \text{CE}(y, \hat{y}_s)$$

**Reasoning for hyperparameters:**
- **T=4.0:** High temperature smooths teacher logits, exposing inter-class similarity information that binary labels cannot convey
- **α=0.5:** Equal weight to soft targets and hard labels balances teacher knowledge transfer with ground truth supervision
- **No class weighting in KD:** Soft targets from the teacher implicitly encode class difficulty; adding explicit class weights would double-count the imbalance adjustment

---

## Key Findings

### 1. Perfect Knowledge Transfer

The distilled student achieves **identical** test metrics to the teacher:
- Same F1 score (0.9645)
- Same confusion matrix pattern (4 FP, 3 FN)
- Same precision-recall balance

### 2. Behavioral Transfer, Not Just Accuracy

| Error Type | Teacher | Baseline | Distilled |
|-----------|---------|----------|-----------|
| False Positives (Ham→Spam) | 4 | 2 | 4 |
| False Negatives (Spam→Ham) | 3 | 5 | 3 |
| **Total Errors** | **7** | **7** | **7** |

The baseline student, trained with class-weighted CE, learns a *conservative* strategy (minimize false positives). The distilled student mimics the teacher's *balanced* strategy. This demonstrates that distillation transfers decision **behavior** — not just accuracy numbers.

### 3. Faster Convergence with Distillation

Despite having more total epochs (15 vs 10), the distilled student reaches its best checkpoint at **epoch 3** vs the baseline's **epoch 6**. Soft targets provide a richer, more informative gradient signal.

### 4. Compression Without Compromise

| Metric | Value |
|--------|-------|
| Parameter reduction | 11.5x (109.5M → 9.5M) |
| Size reduction | 91.3% (417.6 MB → 36.2 MB) |
| F1 retention | 100.00% |
| Efficiency gain | 11.5x better F1/MB ratio |

---

## Project Structure

```
├── data/
│   ├── raw/                        # Original SMS Spam Collection
│   └── processed/                  # Preprocessed train/val/test CSVs
│       ├── train.csv               # 3,617 samples
│       ├── val.csv                 # 776 samples
│       └── test.csv                # 776 samples
├── notebooks/
│   ├── exploration.ipynb           # Comprehensive EDA with plots
│   └── evaluation.ipynb            # Model comparison & statistical analysis
├── preprocessing/
│   ├── step_01_load_raw.py         # Load and parse raw data
│   ├── step_02_deduplicate.py      # Remove 403 duplicates
│   ├── step_03_normalize.py        # URL/phone/whitespace normalization
│   ├── step_04_encode_labels.py    # Label encoding (ham=0, spam=1)
│   ├── step_05_split.py            # Stratified 70/15/15 split
│   ├── step_06_validate.py         # Leakage check
│   ├── step_07_save.py             # Save to CSV
│   ├── step_08_stats.py            # Generate statistics
│   └── DECISIONS.md                # Preprocessing reasoning document
├── src/
│   ├── models/
│   │   ├── teacher_model.py        # BERT-base wrapper (109.5M params)
│   │   └── student_model.py        # Mini-BERT architecture (9.5M params)
│   ├── training/
│   │   ├── train_teacher.py        # Teacher fine-tuning script
│   │   ├── train_student.py        # Student baseline (CE loss)
│   │   └── train_distillation.py   # Knowledge distillation (KL + CE)
│   └── utils/
│       └── training_logs.py        # JSON logging utilities
├── outputs/
│   ├── checkpoints/                # Best model checkpoints
│   │   ├── teacher_best/
│   │   ├── student_baseline_best/
│   │   └── student_distilled_best/
│   ├── plots/                      # Generated visualizations
│   └── logs/                       # Training result JSONs
├── requirements.txt
└── README.md
```

---

## Setup & Reproduction

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (training was done on GPU; inference works on CPU)
- ~2 GB disk space for models and data

### Installation

```bash
# Clone the repository
git clone https://github.com/mrkderchef/on-device-spam-classifier.git
cd on-device-spam-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# .\venv\Scripts\activate       # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Running the Preprocessing Pipeline

```bash
# Run all preprocessing steps sequentially
python preprocessing/step_01_load_raw.py
python preprocessing/step_02_deduplicate.py
python preprocessing/step_03_normalize.py
python preprocessing/step_04_encode_labels.py
python preprocessing/step_05_split.py
python preprocessing/step_06_validate.py
python preprocessing/step_07_save.py
python preprocessing/step_08_stats.py
```

### Training Models

```bash
# 1. Fine-tune teacher (requires GPU, ~30 min)
python src/training/train_teacher.py

# 2. Train student baseline (requires GPU, ~15 min)
python src/training/train_student.py

# 3. Train distilled student (requires GPU + teacher checkpoint, ~20 min)
python src/training/train_distillation.py
```

### Running Notebooks

```bash
jupyter notebook notebooks/exploration.ipynb    # Data analysis
jupyter notebook notebooks/evaluation.ipynb     # Results comparison
```

---

## Design Decisions & Reasoning

### Why max_length=128?

Analysis of BERT tokenization showed that 99.8% of SMS messages fit within 128 tokens. Using 128 instead of BERT's default 512:
- Reduces computation by 4x (attention is O(n²))
- No meaningful information loss
- Matches the short-text nature of SMS

### Why stratified splitting?

With only 12.6% spam messages, random splitting could produce uneven class distributions across splits. Stratification guarantees identical spam ratios in train/val/test, ensuring:
- Unbiased evaluation metrics
- Stable F1 scores across splits
- No lucky/unlucky splits affecting conclusions

### Why class-weighted loss for teacher & baseline?

Without class weights, models achieve high accuracy by simply predicting "ham" for everything (87.4% baseline). Class weights (spam weight ≈ 7x ham weight) force the model to actually learn spam patterns.

### Why NO class weights for distillation?

The teacher's soft targets already encode class difficulty. A confident "ham" prediction of [0.99, 0.01] inherently provides less loss signal than an uncertain one [0.7, 0.3]. Adding explicit class weights on top would over-correct.

### Why T=4.0?

Temperature controls how much probability mass is redistributed:
- T=1: Original sharp distribution → student just sees hard labels
- T=4: Smoothed distribution → student learns from tail probabilities
- T>8: Too smooth → all information is washed out

T=4 is the sweet spot where inter-class relationships are visible but dominant predictions still guide learning.

---

## Limitations & Future Work

### Current Limitations

1. **Single dataset**: Results are demonstrated on SMS Spam Collection only
2. **No mobile benchmark**: Inference speed not measured on actual mobile hardware
3. **No quantization**: Model could be further compressed with INT8/INT4 quantization
4. **English only**: Not tested on multilingual spam

### Potential Extensions

- **Quantization**: INT8 would reduce to ~9 MB, INT4 to ~4.5 MB
- **ONNX export**: For cross-platform mobile deployment
- **CoreML / TensorFlow Lite conversion**: Direct mobile framework integration
- **Multi-language expansion**: Fine-tune on multilingual spam datasets
- **Adversarial testing**: Evaluate robustness against spam obfuscation techniques
- **A/B testing on device**: Real-world deployment metrics

---

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
scipy>=1.10.0
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{on-device-spam-classifier,
  title={Knowledge Distillation for On-Device Spam Detection},
  author={mrkderchef},
  year={2026},
  url={https://github.com/mrkderchef/on-device-spam-classifier}
}
```

## License

MIT
