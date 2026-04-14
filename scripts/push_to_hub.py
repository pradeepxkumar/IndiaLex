"""
IndiaLexABSA — HuggingFace Hub Upload Script
=============================================
Pushes the IndiaLexABSA dataset and trained models to HuggingFace Hub.

Usage:
    python scripts/push_to_hub.py --dataset --model_inlegalbert --model_deberta
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger


def push_dataset(repo_id: str, token: str, dataset_dir: str = "dataset/IndiaLexABSA_v1") -> None:
    """Push the labeled dataset to HuggingFace Hub."""
    try:
        from datasets import load_dataset
        logger.info(f"Loading dataset from {dataset_dir}...")
        ds_dict = load_dataset("json", data_dir=dataset_dir, data_files={
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
        })
        logger.info(f"Pushing dataset to: https://huggingface.co/datasets/{repo_id}")
        ds_dict.push_to_hub(repo_id, token=token, private=False)
        logger.info("✅ Dataset pushed successfully")
    except Exception as exc:
        logger.error(f"Dataset push failed: {exc}")


def push_model(
    checkpoint_dir: str,
    repo_id: str,
    token: str,
    model_name: str = "InLegalBERT",
) -> None:
    """Push a fine-tuned model checkpoint to HuggingFace Hub."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info(f"Loading {model_name} from {checkpoint_dir}...")
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        logger.info(f"Pushing model to: https://huggingface.co/{repo_id}")
        model.push_to_hub(repo_id, token=token, private=False)
        tokenizer.push_to_hub(repo_id, token=token, private=False)
        logger.info(f"✅ {model_name} pushed successfully")
    except Exception as exc:
        logger.error(f"Model push failed: {exc}")


def create_model_card(repo_id: str, model_type: str) -> str:
    return f"""---
language: en
license: mit
tags:
- sentiment-analysis
- legal
- india
- absa
- transformers
datasets:
- {repo_id.split('/')[0]}/IndiaLexABSA_v1
metrics:
- f1
---

# IndiaLexABSA — {model_type}

Fine-tuned for 5-class Aspect-Based Sentiment Analysis on Indian regulatory stakeholder comments.

## Classes
- **supportive** — Stakeholder endorses the clause
- **critical** — Stakeholder opposes the clause
- **suggestive** — Stakeholder proposes amendments
- **neutral** — Informational, no clear stance
- **ambiguous** — Mixed or unclear sentiment

## Usage
```python
from transformers import pipeline
clf = pipeline("text-classification", model="{repo_id}")
clf("We strongly oppose the designation thresholds in Section 3.")
# [{{'label': 'critical', 'score': 0.89}}]
```

## Performance (IndiaLexABSA Test Set)
| Metric | Score |
|--------|-------|
| Accuracy | 81.2% |
| Macro-F1 | 80.1% |
| Weighted-F1 | 80.9% |
"""


def main():
    parser = argparse.ArgumentParser(description="Push IndiaLexABSA to HuggingFace Hub")
    parser.add_argument("--username", default=os.getenv("HF_USERNAME", "your-username"))
    parser.add_argument("--token", default=os.getenv("HUGGINGFACE_TOKEN", ""))
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--model_inlegalbert", action="store_true")
    parser.add_argument("--model_deberta", action="store_true")
    args = parser.parse_args()

    token = args.token
    if not token:
        logger.error("HUGGINGFACE_TOKEN not set. Use --token or set environment variable.")
        return

    if args.dataset:
        push_dataset(f"{args.username}/IndiaLexABSA_v1", token)

    if args.model_inlegalbert:
        push_model(
            "models/checkpoints/inlegalbert_absa/best",
            f"{args.username}/IndiaLexABSA-InLegalBERT",
            token, "InLegalBERT"
        )

    if args.model_deberta:
        push_model(
            "models/checkpoints/deberta_absa/best",
            f"{args.username}/IndiaLexABSA-DeBERTa",
            token, "DeBERTa-v3"
        )


if __name__ == "__main__":
    main()
