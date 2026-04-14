"""
IndiaLexABSA — Ensemble Predictor
=====================================
Combines InLegalBERT + DeBERTa via soft-vote ensemble.
Uses only FREE open-source HuggingFace models — no paid APIs.

Inference logic:
  1. Get probabilities from InLegalBERT
  2. Get probabilities from DeBERTa
  3. Soft-vote: weighted average of probability vectors
  4. If max(combined_prob) < threshold → mark as low-confidence
  5. Return final prediction with source metadata
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger


LABELS = ["supportive", "critical", "suggestive", "neutral", "ambiguous"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

CONFIDENCE_THRESHOLD = 0.65
INLEGALBERT_WEIGHT = 0.5
DEBERTA_WEIGHT = 0.5


class EnsemblePredictor:
    """
    Soft-vote ensemble: InLegalBERT + DeBERTa-v3.
    100% free — uses only open-source HuggingFace models.
    Designed for lazy loading — models are loaded only when first needed.
    """

    def __init__(
        self,
        inlegalbert_checkpoint: str = "models/checkpoints/inlegalbert_absa/best",
        deberta_checkpoint: str = "models/checkpoints/deberta_absa/best",
        inlegalbert_weight: float = INLEGALBERT_WEIGHT,
        deberta_weight: float = DEBERTA_WEIGHT,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        demo_mode: bool = False,
    ):
        self.inlegalbert_ckpt = inlegalbert_checkpoint
        self.deberta_ckpt = deberta_checkpoint
        self.inlegalbert_weight = inlegalbert_weight
        self.deberta_weight = deberta_weight
        self.confidence_threshold = confidence_threshold
        self.demo_mode = demo_mode

        self._inlegalbert = None
        self._deberta = None

        # Stats tracking
        self.stats = {
            "total": 0,
            "inlegalbert_decisive": 0,
            "deberta_decisive": 0,
            "ensemble": 0,
            "low_confidence": 0,
        }

    # ─── Lazy model loaders ───────────────────────────────────────────────────

    def _get_inlegalbert(self):
        if self._inlegalbert is None and not self.demo_mode:
            try:
                from models.inlegalbert_absa import InLegalBERTABSA
                model = InLegalBERTABSA(checkpoint_dir=self.inlegalbert_ckpt)
                model._load_from_checkpoint()
                self._inlegalbert = model
                logger.info("InLegalBERT loaded into ensemble")
            except Exception as exc:
                logger.warning(f"InLegalBERT unavailable: {exc}")
        return self._inlegalbert

    def _get_deberta(self):
        if self._deberta is None and not self.demo_mode:
            try:
                from models.deberta_absa import DeBERTaABSA
                model = DeBERTaABSA(checkpoint_dir=self.deberta_ckpt)
                model._load_from_checkpoint()
                self._deberta = model
                logger.info("DeBERTa loaded into ensemble")
            except Exception as exc:
                logger.warning(f"DeBERTa unavailable: {exc}")
        return self._deberta

    # ─── Demo mode ────────────────────────────────────────────────────────────

    def _demo_predict(self, sentence: str) -> dict:
        """Deterministic demo prediction based on keyword heuristics."""
        import random
        s_lower = sentence.lower()
        if any(w in s_lower for w in ["oppose", "reject", "against", "critical", "concern", "object", "strongly disagree"]):
            label = "critical"
        elif any(w in s_lower for w in ["support", "welcome", "appreciate", "agree", "endorse", "commend"]):
            label = "supportive"
        elif any(w in s_lower for w in ["suggest", "recommend", "propose", "amend", "revise", "consider"]):
            label = "suggestive"
        elif any(w in s_lower for w in ["define", "states", "provides", "establishes", "sets out"]):
            label = "neutral"
        else:
            label = random.choice(LABELS)

        probs = {l: 0.05 for l in LABELS}
        probs[label] = 0.70
        remaining = (1.0 - 0.70 - 0.05 * 4) / (len(LABELS) - 1)
        for l in LABELS:
            if l != label:
                probs[l] = remaining + 0.05

        return {
            "label": label,
            "confidence": 0.70,
            "probabilities": probs,
            "source": "demo",
        }

    # ─── Core prediction ──────────────────────────────────────────────────────

    def predict_one(
        self,
        sentence: str,
        clause_context: str = "",
    ) -> dict:
        """Predict sentiment for a single sentence."""
        self.stats["total"] += 1

        # Demo mode
        if self.demo_mode:
            return self._demo_predict(sentence)

        il_model = self._get_inlegalbert()
        db_model = self._get_deberta()

        il_probs = None
        db_probs = None

        if il_model:
            try:
                result = il_model.predict([sentence], [clause_context])[0]
                il_probs = np.array([result["probabilities"][l] for l in LABELS])
            except Exception as exc:
                logger.warning(f"InLegalBERT inference failed: {exc}")

        if db_model:
            try:
                result = db_model.predict([sentence], [clause_context])[0]
                db_probs = np.array([result["probabilities"][l] for l in LABELS])
            except Exception as exc:
                logger.warning(f"DeBERTa inference failed: {exc}")

        # Combine probabilities
        if il_probs is not None and db_probs is not None:
            combined = self.inlegalbert_weight * il_probs + self.deberta_weight * db_probs
            source = "ensemble"
            self.stats["ensemble"] += 1
        elif il_probs is not None:
            combined = il_probs
            source = "inlegalbert"
            self.stats["inlegalbert_decisive"] += 1
        elif db_probs is not None:
            combined = db_probs
            source = "deberta"
            self.stats["deberta_decisive"] += 1
        else:
            # Both models unavailable — use demo fallback
            return self._demo_predict(sentence)

        max_conf = float(combined.max())
        label_idx = int(combined.argmax())
        label = ID2LABEL[label_idx]

        # Track low-confidence predictions
        if max_conf < self.confidence_threshold:
            self.stats["low_confidence"] += 1

        return {
            "label": label,
            "confidence": max_conf,
            "probabilities": {ID2LABEL[i]: float(p) for i, p in enumerate(combined)},
            "source": source,
            "low_confidence": max_conf < self.confidence_threshold,
        }

    def predict_batch(self, sentences: list[str], clause_contexts: Optional[list[str]] = None) -> list[dict]:
        """Predict sentiment for a batch of sentences."""
        if clause_contexts is None:
            clause_contexts = [""] * len(sentences)

        from tqdm import tqdm
        return [
            self.predict_one(sent, ctx)
            for sent, ctx in tqdm(
                zip(sentences, clause_contexts),
                total=len(sentences),
                desc="Ensemble inference",
            )
        ]

    def get_stats(self) -> dict:
        total = self.stats["total"]
        if total == 0:
            return self.stats
        return {
            **self.stats,
            "low_confidence_pct": f"{self.stats['low_confidence'] / total * 100:.1f}%",
        }
