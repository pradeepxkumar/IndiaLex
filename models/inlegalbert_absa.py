"""
InLegalBERT ABSA model wrapper.
Loads the fine-tuned InLegalBERT checkpoint for inference.
"""
from __future__ import annotations

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
import numpy as np


class InLegalBERTABSA:
    """Wrapper for fine-tuned InLegalBERT sentiment classifier."""

    def __init__(self, checkpoint_dir: str = "models/checkpoints/inlegalbert_absa/best"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def _load_from_checkpoint(self):
        if self.model is not None:
            return
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.checkpoint_dir)
        ).to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        logger.info(f"InLegalBERT loaded from {self.checkpoint_dir}")

    def predict(self, sentences: list[str], contexts: list[str] = None) -> list[dict]:
        if self.model is None:
            self._load_from_checkpoint()

        results = []
        for sent in sentences:
            inputs = self.tokenizer(
                sent, truncation=True, max_length=256,
                padding=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            label_idx = int(probs.argmax())

            results.append({
                "label": self.id2label[label_idx],
                "confidence": float(probs.max()),
                "probabilities": {self.id2label[i]: float(p) for i, p in enumerate(probs)},
            })

        return results
