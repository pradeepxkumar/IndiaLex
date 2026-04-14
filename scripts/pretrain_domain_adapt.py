"""
IndiaLexABSA — Domain Adaptation Pre-trainer
=============================================
Continues pre-training InLegalBERT on raw stakeholder PDF text
via Masked Language Modeling (MLM). This adapts the model to
the specific vocabulary of MCA consultations before fine-tuning.

Expected improvement: +2-4 macro-F1 points on downstream ABSA.

Usage:
    python scripts/pretrain_domain_adapt.py \
        --text_dir data/processed/ \
        --output_dir models/checkpoints/inlegalbert_domain_adapted/ \
        --epochs 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger


def collect_corpus(text_dir: str, max_chars: int = 5_000_000) -> str:
    """Collect all extracted text from processed documents."""
    text_dir = Path(text_dir)
    corpus_parts = []
    total_chars = 0

    for json_path in sorted(text_dir.glob("*.json")):
        try:
            with open(json_path, encoding="utf-8") as f:
                doc = json.load(f)
            text = doc.get("full_text", "")
            if text:
                corpus_parts.append(text)
                total_chars += len(text)
                if total_chars >= max_chars:
                    break
        except Exception:
            pass

    corpus = "\n\n".join(corpus_parts)
    logger.info(f"Corpus collected: {total_chars:,} chars from {len(corpus_parts)} documents")
    return corpus


def run_domain_adaptation(
    text_dir: str,
    output_dir: str,
    base_model: str = "law-ai/InLegalBERT",
    num_epochs: int = 3,
    batch_size: int = 8,
    mlm_probability: float = 0.15,
    max_seq_length: int = 512,
) -> None:
    """Run Masked Language Modeling domain adaptation."""
    try:
        import torch
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset
    except ImportError as exc:
        logger.error(f"Missing dependency: {exc}. Run: pip install transformers datasets torch")
        return

    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)

    # Collect corpus
    corpus = collect_corpus(text_dir)
    if not corpus.strip():
        logger.error("No corpus text found. Check text_dir.")
        return

    # Tokenize
    logger.info("Tokenizing corpus...")
    sentences = [s.strip() for s in corpus.split("\n") if len(s.strip()) > 20][:50_000]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    dataset = Dataset.from_dict({"text": sentences})
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Split
    split = tokenized.train_test_split(test_size=0.05, seed=42)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        run_name="inlegalbert-domain-adapt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info(f"Starting domain adaptation: {len(split['train'])} train, {len(split['test'])} eval sentences")
    trainer.train()

    # Save adapted model
    output_dir = Path(output_dir) / "adapted_model"
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Domain-adapted model saved → {output_dir}")
    logger.info("Use this as --model_id in inlegalbert_absa.py for fine-tuning")


def main():
    parser = argparse.ArgumentParser(description="MLM domain adaptation pre-training")
    parser.add_argument("--text_dir", default="data/processed/")
    parser.add_argument("--output_dir", default="models/checkpoints/inlegalbert_domain_adapted/")
    parser.add_argument("--base_model", default="law-ai/InLegalBERT")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    args = parser.parse_args()

    run_domain_adaptation(
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_prob,
    )


if __name__ == "__main__":
    main()
