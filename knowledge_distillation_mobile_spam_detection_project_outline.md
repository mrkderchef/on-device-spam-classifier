# Knowledge Distillation for On-Device Spam Detection

## Official Topic Category
Model Compression: Distillation of Large Language Models

---

# Project Title
## Knowledge Distillation for On-Device Spam Detection

---

# 1. Core Project Idea

The goal of this project is to investigate whether knowledge distillation can compress large transformer-based language models into lightweight student models that are suitable for theoretical on-device deployment on modern smartphones such as the iPhone 16.

The project focuses on spam and phishing text detection as a realistic mobile NLP security use case.

Instead of building a generic chatbot or training a massive LLM from scratch, the project concentrates on practical and deployment-oriented NLP compression.

The central research question is:

> Can a lightweight distilled transformer model retain strong spam/phishing detection performance while significantly reducing computational and memory requirements for mobile deployment?

---

# 2. Story and Motivation

The project narrative is extremely important.

The work is NOT framed as:

> “We compress models because compression is interesting.”

Instead, the story is:

> Modern transformer language models achieve excellent NLP performance, but are often too computationally expensive for mobile and edge deployment. Through knowledge distillation, we investigate whether a smaller student model can inherit much of the teacher model’s performance while becoming theoretically deployable on mobile devices.

The use case is realistic:

- SMS spam detection
- phishing detection
- mobile security filtering
- privacy-preserving inference
- offline capability
- low latency edge inference

The project should argue that many applications benefit from local inference:

- sensitive user data remains on device
- lower latency
- offline functionality
- reduced cloud infrastructure cost
- reduced bandwidth requirements

The project does NOT claim:

> “Cloud inference is obsolete.”

Instead, the project investigates the tradeoff between:

- model performance
- computational cost
- deployment feasibility

---

# 3. Dataset Selection

## Main Dataset
### SMS Spam Collection Dataset

Source:
- UCI Machine Learning Repository

Dataset Characteristics:
- ~5,500 SMS messages
- binary classification
- labels:
  - spam
  - ham (non-spam)

Why this dataset was selected:

- directly mobile relevant
- realistic smartphone use case
- small enough for controlled experimentation
- simple preprocessing
- well-known benchmark dataset
- fast iteration speed
- ideal for distillation experiments

Example messages:

Ham:
> “Hey are we still meeting tomorrow?”

Spam:
> “Congratulations! You won a free iPhone. Click here now.”

---

# 4. Problem Formulation

The problem is formulated as a text classification task.

Input:
- SMS message text

Output:
- probability distribution over:
  - spam
  - ham

Teacher model:
- large transformer classifier

Student model:
- smaller transformer classifier

Goal:
- retain strong detection capability
- reduce memory and compute requirements

---

# 5. Planned Model Architecture

## Teacher Model

The teacher model represents the larger, more powerful NLP model.

Possible teacher models:

- BERT-base
- DistilBERT

Most likely choice:
### BERT-base

Reason:
- strong benchmark model
- widely used
- suitable for text classification
- realistic transformer-based NLP architecture
- sufficiently large to demonstrate compression benefits

Approximate characteristics:

| Property | Teacher |
|---|---|
| Parameters | ~110M |
| Layers | 12 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| FP32 Size | ~420MB |

---

## Student Model

The student model is intentionally smaller and more efficient.

The student architecture will likely:

- reduce transformer layers
- reduce hidden dimension size
- reduce attention heads

Example configuration:

| Property | Student |
|---|---|
| Parameters | ~5M–15M |
| Layers | 2–4 |
| Hidden Size | 256–384 |
| Attention Heads | 4 |
| FP32 Size | ~20–60MB |

The student model should theoretically fit mobile deployment constraints much better.

---

# 6. Distillation Process

## Training Pipeline

### Step 1 — Teacher Fine-Tuning

The teacher model is fine-tuned on the SMS spam dataset.

Goal:
- achieve strong classification performance
- generate meaningful soft targets

The teacher outputs probability distributions instead of only hard labels.

Example:

| Class | Probability |
|---|---|
| Spam | 0.82 |
| Ham | 0.18 |

These probability distributions contain “dark knowledge”.

The student learns not only the correct class, but also the teacher’s uncertainty structure.

---

### Step 2 — Student Baseline Training

The student model is trained normally using standard cross entropy loss.

This creates a baseline for comparison.

---

### Step 3 — Knowledge Distillation Training

The student model is trained using:

- hard labels
- teacher soft targets

The goal is to transfer knowledge from the teacher to the student.

---

# 7. Theoretical Foundation

The report should include a strong theoretical section.

---

## 7.1 Transformer Models

The report should explain:

- token embeddings
- positional embeddings
- self-attention
- transformer encoder architecture
- classification head

Important concepts:

- queries
- keys
- values
- attention scores
- contextual representations

---

## 7.2 Knowledge Distillation

The report should explain:

- teacher-student paradigm
- hard targets vs soft targets
- dark knowledge
- temperature scaling
- KL divergence

---

## 7.3 Softmax with Temperature

The report should mathematically derive and explain:

\[
p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
\]

Explanation:

- T = temperature
- higher temperature softens the probability distribution
- softened distributions reveal class relationships

---

## 7.4 Distillation Loss

The report should explain and derive:

\[
L = \alpha T^2 KL(p_t || p_s) + (1-\alpha)L_{CE}
\]

Where:

- KL = Kullback-Leibler divergence
- teacher probabilities guide the student
- cross entropy preserves correct labels
- alpha balances both objectives

---

## 7.5 Dark Knowledge

Important conceptual explanation:

Hard labels:

| Label |
|---|
| Spam |

Teacher probabilities:

| Class | Probability |
|---|---|
| Spam | 0.82 |
| Ham | 0.18 |

The teacher communicates:
- uncertainty
- semantic similarity
- boundary information

This additional information helps the student generalize better.

---

# 8. Mathematical Example Section

The report should later include a worked numerical example using a real dataset sample.

Example idea:

SMS:
> “Congratulations! Claim your free reward now!”

Teacher logits:

| Class | Logit |
|---|---|
| Spam | 4.2 |
| Ham | 1.1 |

Using temperature scaling:

\[
p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
\]

The report should manually calculate:

- softmax outputs
- softened probabilities
- KL divergence contribution
- combined loss

This section should demonstrate mathematically how knowledge transfer occurs.

---

# 9. Planned Experiments

## Experiment 1 — Teacher Performance

Evaluate:

- accuracy
- precision
- recall
- F1 score

---

## Experiment 2 — Student Baseline

Train student without distillation.

Compare:
- performance drop
- parameter reduction

---

## Experiment 3 — Distilled Student

Train student with knowledge distillation.

Measure:
- performance recovery
- efficiency improvements

---

## Experiment 4 — Temperature Sweep

Evaluate multiple temperatures:

- T = 1
- T = 2
- T = 4
- T = 8

Goal:
- investigate how softened probability distributions affect learning.

---

## Experiment 5 — Compression Comparison

Compare:

| Model | Params | Model Size | Accuracy |
|---|---|---|---|
| Teacher | Large | Large | Highest |
| Student | Small | Small | Lower |
| KD Student | Small | Small | Improved |

---

# 10. Hardware and Deployment Comparison

The project should include a theoretical deployment feasibility analysis.

Important:

The project does NOT actually deploy to iPhone.

Instead, it investigates whether deployment would theoretically be realistic.

---

## Teacher Hardware Characteristics

The teacher model is assumed to require:

- high memory usage
- large storage
- higher inference cost
- cloud/server-oriented deployment

Possible environment:
- GPU server
- desktop GPU
- cloud inference

---

## Student Hardware Characteristics

The student model should theoretically target:

- mobile CPUs
- mobile NPUs
- edge inference
- low-memory environments

Discussion topics:

- theoretical RAM requirements
- parameter count
- FP32 vs FP16 vs INT8
- inference latency
- energy efficiency
- CoreML/ONNX compatibility

---

## iPhone 16 Discussion

The report should briefly discuss:

- Apple Neural Engine
- on-device AI trend
- edge inference
- mobile AI acceleration

The work should argue that:

- the student model could theoretically fit within realistic mobile deployment constraints
- while the teacher model would be significantly less practical

---

# 11. Evaluation Metrics

The report should evaluate:

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

## Efficiency Metrics

- Parameter count
- Estimated memory usage
- FLOPs
- Inference speed
- Compression ratio

---

# 12. Expected Results

Expected outcome:

- the student baseline performs noticeably worse than the teacher
- the distilled student recovers much of the lost performance
- the distilled student remains significantly smaller and more efficient

Expected narrative:

> Knowledge distillation enables efficient mobile-friendly NLP models while preserving a large portion of transformer model performance.

---

# 13. Planned Repository Structure

The repository should be clean and research-oriented.

---

## Proposed Structure

```text
project-root/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── exploration.ipynb
│   ├── evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── dataset.py
│   │
│   ├── models/
│   │   ├── teacher_model.py
│   │   ├── student_model.py
│   │
│   ├── training/
│   │   ├── train_teacher.py
│   │   ├── train_student.py
│   │   ├── train_distillation.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │
│   ├── utils/
│
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   ├── logs/
│
├── reports/
│
├── requirements.txt
├── README.md
```

---

# 14. Repository Goals

The repository should support:

- reproducible experiments
- clean training pipelines
- easy comparison between teacher/student models
- automated evaluation
- visualization generation
- export of metrics and plots

---

# 15. Potential Visualizations

The report should contain:

- accuracy comparison plots
- parameter count comparison
- memory comparison
- confusion matrices
- temperature sweep plots
- compression ratio plots
- inference speed comparison

---

# 16. Final Intended Conclusion

The intended conclusion of the work is:

> Knowledge distillation provides a practical pathway toward lightweight transformer-based NLP systems suitable for edge and mobile deployment.

The project aims to show that:

- substantial compression is possible
- performance can largely be retained
- mobile NLP applications are realistic
- distillation is an effective compression strategy for transformer models

---

# 17. Important Scope Limitations

The project explicitly does NOT aim to:

- train a massive LLM from scratch
- reproduce GPT-scale systems
- perform real iPhone deployment
- build a production-ready security application

The project instead focuses on:

- controlled transformer distillation experiments
- efficiency analysis
- deployment-oriented NLP research
- mobile feasibility investigation

---

# 18. Overall Research Contribution

The contribution of the project is not:

> “We invented knowledge distillation.”

Instead, the contribution is:

> investigating whether distilled transformer models can realistically support efficient on-device spam detection while maintaining strong classification performance.

