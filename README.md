# Knowledge Distillation for On-Device Spam Detection

## Overview

This project investigates whether **knowledge distillation** can compress large transformer-based language models into lightweight student models suitable for theoretical on-device deployment on modern smartphones (e.g., iPhone 16).

The focus is on **SMS spam and phishing text detection** as a realistic mobile NLP security use case.

### Research Question

> Can a lightweight distilled transformer model retain strong spam/phishing detection performance while significantly reducing computational and memory requirements for mobile deployment?

## Motivation

Modern transformer models achieve excellent NLP performance but are often too computationally expensive for mobile and edge deployment. This project investigates the tradeoff between:

- **Model performance** vs. computational cost
- **Privacy-preserving** on-device inference
- **Offline capability** and low latency
- Reduced cloud infrastructure and bandwidth requirements

## Dataset

**SMS Spam Collection Dataset** from the UCI Machine Learning Repository:
- ~5,500 SMS messages
- Binary classification: `spam` / `ham`
- Directly mobile-relevant and well-suited for controlled distillation experiments

## Architecture

| Property | Teacher (BERT-base) | Student |
|---|---|---|
| Parameters | ~110M | ~5M–15M |
| Layers | 12 | 2–4 |
| Hidden Size | 768 | 256–384 |
| Attention Heads | 12 | 4 |
| FP32 Size | ~420MB | ~20–60MB |

## Distillation Process

1. **Teacher Fine-Tuning** — Fine-tune BERT-base on SMS spam dataset to generate soft targets
2. **Student Baseline** — Train student model with standard cross-entropy loss
3. **Knowledge Distillation** — Train student using both hard labels and teacher soft targets

The distillation loss combines KL divergence and cross-entropy:

$$L = \alpha T^2 \text{KL}(p_t \| p_s) + (1-\alpha) L_{CE}$$

## Experiments

1. **Teacher Performance** — Baseline accuracy, precision, recall, F1
2. **Student Baseline** — Performance without distillation
3. **Distilled Student** — Performance with knowledge distillation
4. **Temperature Sweep** — T ∈ {1, 2, 4, 8}
5. **Compression Comparison** — Parameter count, model size, accuracy tradeoffs

## Project Structure

```
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Preprocessed data
├── notebooks/
│   ├── exploration.ipynb       # Data exploration & analysis
│   └── evaluation.ipynb        # Results evaluation & visualization
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Text preprocessing pipeline
│   │   └── dataset.py          # PyTorch dataset classes
│   ├── models/
│   │   ├── teacher_model.py    # BERT-based teacher model
│   │   └── student_model.py    # Lightweight student model
│   ├── training/
│   │   ├── train_teacher.py    # Teacher fine-tuning
│   │   ├── train_student.py    # Student baseline training
│   │   └── train_distillation.py # Knowledge distillation training
│   ├── evaluation/
│   │   ├── metrics.py          # Classification & efficiency metrics
│   │   └── benchmark.py        # Model benchmarking utilities
│   └── utils/                  # Shared utilities
├── outputs/
│   ├── checkpoints/            # Model checkpoints
│   ├── plots/                  # Generated visualizations
│   └── logs/                   # Training logs
├── reports/                    # Report documents
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/mrkderchef/on-device-spam-classifier.git
cd on-device-spam-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Scope

This project focuses on controlled transformer distillation experiments and theoretical mobile deployment feasibility. It does **not** include actual device deployment, production-ready applications, or GPT-scale model training.

## License

MIT
