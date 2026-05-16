"""
Preprocessing package for SMS Spam Detection Knowledge Distillation project.

This folder contains the complete data preparation pipeline, split into
individual steps for clarity and storytelling. Each file is self-contained
and documents WHY specific decisions were made.

Steps:
  01 - Load and inspect raw data
  02 - Clean duplicates
  03 - Text normalization (minimal, transformer-aware)
  04 - Label encoding (string → integer)
  05 - Train/Val/Test stratified split
  06 - Tokenization analysis (max_length decision)
  07 - Class imbalance analysis and strategy
  08 - Full pipeline execution

Run the full pipeline:
  python preprocessing/step_08_run_full_pipeline.py
"""
