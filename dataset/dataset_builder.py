"""
IndiaLexABSA — Dataset Builder
==================================
Merges silver (GPT) labels with human annotations, applies quality
filters, creates train/val/test splits, and pushes to HuggingFace Hub.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


LABELS = ["supportive", "critical", "suggestive", "neutral", "ambiguous"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

DATASET_DIR = "dataset/IndiaLexABSA_v1"


class DatasetBuilder:
    """Builds HuggingFace-compatible dataset from labeled triples."""

    def __init__(
        self,
        silver_path: str = "data/processed/silver_labeled.jsonl",
        human_db_path: str = "data/processed/human_annotations.jsonl",
        output_dir: str = DATASET_DIR,
        min_confidence: int = 3,
        agreement_only: bool = False,
    ):
        self.silver_path = Path(silver_path)
        self.human_path = Path(human_db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_confidence = min_confidence
        self.agreement_only = agreement_only

    def _load_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return items

    def _quality_filter(self, items: list[dict]) -> list[dict]:
        """Apply quality filters to remove low-quality labels."""
        filtered = []
        for item in items:
            label = item.get("label")
            confidence = item.get("confidence", 0)

            if label not in LABELS:
                continue
            if isinstance(confidence, int) and confidence < self.min_confidence:
                continue
            if not item.get("sentence") or len(item["sentence"].strip()) < 10:
                continue
            if not item.get("clause_id"):
                continue
            filtered.append(item)
        return filtered

    def _normalize(self, item: dict) -> dict:
        """Normalize a triple to a consistent schema."""
        return {
            "sent_id": item.get("sent_id", ""),
            "sentence": item.get("sentence", item.get("text", "")),
            "clause_id": item.get("clause_id", ""),
            "clause_title": item.get("clause_title", ""),
            "clause_text": item.get("clause_text", ""),
            "clause_summary": item.get("clause_summary", ""),
            "submitter": item.get("submitter", "Unknown"),
            "category": item.get("category", "individual"),
            "language": item.get("sentence_lang", item.get("language", "en")),
            "was_translated": item.get("was_translated", False),
            "label": item.get("label"),
            "label_id": LABEL2ID.get(item.get("label", ""), -1),
            "confidence": item.get("confidence", 3),
            "label_source": item.get("label_source", "gpt"),
            "similarity_score": item.get("similarity_score"),
        }

    def merge(self) -> list[dict]:
        """Merge silver + human labels, with human labels taking precedence."""
        silver = self._load_jsonl(self.silver_path)
        human = self._load_jsonl(self.human_path)

        logger.info(f"Silver labels: {len(silver)}, Human labels: {len(human)}")

        # Build lookup by sent_id
        merged: dict[str, dict] = {}
        for item in silver:
            norm = self._normalize(item)
            if norm["sent_id"]:
                merged[norm["sent_id"]] = norm

        # Human labels override silver labels
        for item in human:
            norm = self._normalize(item)
            if norm["sent_id"]:
                norm["label_source"] = "human"
                merged[norm["sent_id"]] = norm

        all_items = list(merged.values())
        filtered = self._quality_filter(all_items)
        logger.info(f"After quality filter: {len(filtered)}/{len(all_items)} items")
        return filtered

    def split(
        self, items: list[dict],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Stratified split by label."""
        labels = [item["label"] for item in items]
        train, temp, y_train, y_temp = train_test_split(
            items, labels,
            test_size=(1 - train_ratio),
            stratify=labels,
            random_state=seed,
        )
        val_ratio_adj = val_ratio / (val_ratio + test_ratio)
        val, test, _, _ = train_test_split(
            temp, y_temp,
            test_size=(1 - val_ratio_adj),
            stratify=y_temp,
            random_state=seed,
        )

        logger.info(f"Split → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        for split_name, split_items in [("Train", train), ("Val", val), ("Test", test)]:
            dist = Counter(i["label"] for i in split_items)
            logger.info(f"  {split_name}: {dict(dist)}")

        return train, val, test

    def save_splits(self, train, val, test) -> None:
        for name, items in [("train", train), ("validation", val), ("test", test)]:
            path = self.output_dir / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved {name}: {len(items)} items → {path}")

        # Save dataset card
        self._save_dataset_card(len(train), len(val), len(test))

    def _save_dataset_card(self, n_train, n_val, n_test) -> None:
        card = f"""---
language: en
license: cc-by-4.0
task_categories:
- text-classification
task_ids:
- sentiment-analysis
tags:
- legal
- india
- absa
- competition-law
- nlp
---

# IndiaLexABSA Dataset

Aspect-Based Sentiment Analysis dataset for Indian regulatory stakeholder comments
on the Digital Competition Bill, 2024.

## Dataset Statistics
- Train: {n_train} examples
- Validation: {n_val} examples
- Test: {n_test} examples
- Total: {n_train + n_val + n_test} examples

## Classes
- **supportive**: Stakeholder endorses the clause
- **critical**: Stakeholder opposes the clause
- **suggestive**: Stakeholder proposes amendments
- **neutral**: Informational, no clear stance
- **ambiguous**: Mixed or unclear sentiment

## Citation
```bibtex
@dataset{{indialexabsa2024,
  title={{IndiaLexABSA: Aspect-Based Sentiment Analysis for Indian Regulatory Comments}},
  year={{2024}},
  license={{cc-by-4.0}}
}}
```
"""
        with open(self.output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(card)

    def push_to_hub(self, repo_id: str, token: Optional[str] = None) -> None:
        """Push dataset to HuggingFace Hub."""
        import os
        token = token or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            logger.warning("HUGGINGFACE_TOKEN not set — skipping Hub upload")
            return
        try:
            from datasets import load_dataset
            ds = load_dataset("json", data_dir=str(self.output_dir))
            ds.push_to_hub(repo_id, token=token)
            logger.info(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")
        except Exception as exc:
            logger.error(f"Hub upload failed: {exc}")

    def build(self, push_to_hub: str = "") -> tuple[list, list, list]:
        """Full build pipeline."""
        items = self.merge()
        train, val, test = self.split(items)
        self.save_splits(train, val, test)
        if push_to_hub:
            self.push_to_hub(push_to_hub)
        return train, val, test
