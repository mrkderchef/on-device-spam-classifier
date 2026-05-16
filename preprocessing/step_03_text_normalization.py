"""
Step 3: Text normalization — standardize text while preserving semantic content.

This is MINIMAL preprocessing because we use BERT-based models:
- BERT has its own WordPiece tokenizer that handles most text variations.
- Heavy preprocessing (stemming, stopword removal) would HURT transformer performance.
- We only normalize aspects that add noise without semantic value.

WHY minimal preprocessing for transformers:
- BERT was pre-trained on raw text — it expects natural language input.
- Subword tokenization handles unknown words gracefully.
- Removing stopwords destroys sentence structure that attention mechanisms use.
- Stemming/lemmatization conflicts with BERT's vocabulary.

WHAT we DO normalize:
1. URLs → [URL] token (the specific URL is irrelevant for spam detection)
2. Phone numbers → [PHONE] token (specific digits are irrelevant)
3. Excessive whitespace → single spaces
4. Strip leading/trailing whitespace

WHAT we DO NOT do (and why):
- No lowercasing: BERT-uncased handles this internally
- No stopword removal: Destroys sentence structure needed for attention
- No stemming/lemmatization: Conflicts with WordPiece tokenizer
- No punctuation removal: Punctuation carries meaning (!!!, ???)
- No number removal: Numbers can indicate spam patterns (e.g., prize amounts)
"""

import re
import pandas as pd
from step_01_load_and_inspect import load_raw_dataset
from step_02_clean_duplicates import remove_duplicates


def normalize_urls(text: str) -> str:
    """Replace URLs with [URL] token.
    
    Rationale: The specific URL is not informative for classification.
    What matters is the PRESENCE of a URL, not its content.
    Spam often contains URLs, but the domain changes constantly.
    """
    return re.sub(r"http\S+|www\.\S+|https\S+", "[URL]", text)


def normalize_phone_numbers(text: str) -> str:
    """Replace phone numbers with [PHONE] token.
    
    Rationale: Specific phone numbers are not generalizable features.
    The presence of a phone number is relevant (common in spam).
    Matches sequences of 7+ digits (with optional separators).
    """
    # Match various phone formats: +44123456789, 0800-123-456, 08001234567
    text = re.sub(r"\+?\d[\d\s\-]{6,}\d", "[PHONE]", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single space.
    
    Rationale: Multiple spaces/tabs/newlines add noise without meaning.
    """
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """Apply all normalization steps in sequence.
    
    Order matters:
    1. URLs first (they contain numbers that would match phone patterns)
    2. Phone numbers second
    3. Whitespace last (cleanup after replacements)
    """
    text = normalize_urls(text)
    text = normalize_phone_numbers(text)
    text = normalize_whitespace(text)
    return text


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: TEXT NORMALIZATION")
    print("=" * 60)
    
    df = load_raw_dataset()
    df = remove_duplicates(df)
    
    # Show before/after examples
    print("\n--- Normalization Examples ---\n")
    
    # Find messages with URLs
    url_msgs = df[df["text"].str.contains(r"http|www\.", regex=True)]
    if len(url_msgs) > 0:
        sample = url_msgs.iloc[0]["text"]
        print(f"BEFORE: {sample[:120]}")
        print(f"AFTER:  {normalize_text(sample)[:120]}")
        print()
    
    # Find messages with phone numbers
    phone_msgs = df[df["text"].str.contains(r"\d{7,}", regex=True)]
    if len(phone_msgs) > 0:
        sample = phone_msgs.iloc[0]["text"]
        print(f"BEFORE: {sample[:120]}")
        print(f"AFTER:  {normalize_text(sample)[:120]}")
        print()
    
    # Apply normalization to full dataset
    df["text_normalized"] = df["text"].apply(normalize_text)
    
    # Statistics on changes
    changed = (df["text"] != df["text_normalized"]).sum()
    print(f"\nMessages modified by normalization: {changed}/{len(df)} ({changed/len(df)*100:.1f}%)")
    
    # Length distribution after normalization
    df["len_before"] = df["text"].str.len()
    df["len_after"] = df["text_normalized"].str.len()
    avg_reduction = (df["len_before"] - df["len_after"]).mean()
    print(f"Average character reduction: {avg_reduction:.1f} chars")
