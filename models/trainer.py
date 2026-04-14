"""
IndiaLexABSA — Unified Trainer
================================
CLI entrypoint for model training. Handles:
  - Domain adaptation (MLM pre-training)
  - Two-step fine-tuning (LOCO → IndiaLexABSA)
  - Single-model fine-tuning

Usage:
    python models/trainer.py --model inlegalbert --config configs/training_config.yaml
    python models/trainer.py --model deberta --config configs/training_config.yaml
    python models/trainer.py --eval_only --checkpoint models/checkpoints/inlegalbert_absa/best
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from loguru import logger


def load_dataset_split(split_path: str) -> list[dict]:
    items = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} items from {split_path}")
    return items


def train_inlegalbert(config: dict) -> None:
    from models.inlegalbert_absa import InLegalBERTABSA
    dataset_dir = Path("dataset/IndiaLexABSA_v1")
    train_data = load_dataset_split(str(dataset_dir / "train.jsonl"))
    val_data = load_dataset_split(str(dataset_dir / "validation.jsonl"))

    tc = config.get("training", {}).get("inlegalbert", {})
    model = InLegalBERTABSA(
        checkpoint_dir=config["training"]["output_dir"] + "/inlegalbert_absa",
        class_weights=config["data"].get("class_weights"),
    )
    model.train(
        train_triples=train_data,
        val_triples=val_data,
        num_epochs=tc.get("num_train_epochs", config["training"]["num_train_epochs"]),
        learning_rate=tc.get("learning_rate", 2e-5),
        batch_size=tc.get("per_device_train_batch_size", 16),
        warmup_ratio=tc.get("warmup_ratio", 0.1),
        weight_decay=tc.get("weight_decay", 0.01),
    )


def train_deberta(config: dict) -> None:
    from models.deberta_absa import DeBERTaABSA
    dataset_dir = Path("dataset/IndiaLexABSA_v1")
    train_data = load_dataset_split(str(dataset_dir / "train.jsonl"))
    val_data = load_dataset_split(str(dataset_dir / "validation.jsonl"))

    tc = config.get("training", {}).get("deberta", {})
    model = DeBERTaABSA(
        checkpoint_dir=config["training"]["output_dir"] + "/deberta_absa",
        class_weights=config["data"].get("class_weights"),
    )
    model.train(
        train_triples=train_data,
        val_triples=val_data,
        num_epochs=config["training"]["num_train_epochs"],
        learning_rate=tc.get("learning_rate", 1.5e-5),
        batch_size=tc.get("per_device_train_batch_size", 8),
        warmup_ratio=tc.get("warmup_ratio", 0.15),
        weight_decay=tc.get("weight_decay", 0.01),
    )


def evaluate_model(checkpoint: str) -> None:
    from models.inlegalbert_absa import InLegalBERTABSA
    from evaluation.metrics import compute_all_metrics, print_classification_report
    dataset_dir = Path("dataset/IndiaLexABSA_v1")
    test_data = load_dataset_split(str(dataset_dir / "test.jsonl"))

    model = InLegalBERTABSA(checkpoint_dir=checkpoint)
    model._load_from_checkpoint(checkpoint)

    sentences = [t.get("sentence","") for t in test_data]
    contexts = [f"{t.get('clause_title','')}: {(t.get('clause_text') or '')[:200]}" for t in test_data]
    y_true = [t.get("label","neutral") for t in test_data]

    results = model.predict(sentences, contexts)
    y_pred = [r["label"] for r in results]

    metrics = compute_all_metrics(y_true, y_pred, model_name="InLegalBERT")
    logger.info(f"Test results:\n{print_classification_report(y_true, y_pred)}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="IndiaLexABSA Model Trainer")
    parser.add_argument("--model", choices=["inlegalbert", "deberta", "both"], default="inlegalbert")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default="models/checkpoints/inlegalbert_absa/best")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.eval_only:
        evaluate_model(args.checkpoint)
        return

    if args.model in ("inlegalbert", "both"):
        logger.info("Training InLegalBERT...")
        train_inlegalbert(config)

    if args.model in ("deberta", "both"):
        logger.info("Training DeBERTa...")
        train_deberta(config)


if __name__ == "__main__":
    main()
