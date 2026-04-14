"""
IndiaLexABSA — SHAP Explainability
=====================================
Token-level attribution for sentiment predictions using SHAP.
Generates highlighted HTML for the dashboard's deep-dive page.

Two backends:
  1. SHAP DeepExplainer (for PyTorch transformer models)
  2. SHAP KernelExplainer (fallback, slower but model-agnostic)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

SHAP_CACHE_DIR = "dashboard/.shap_cache"

SENTIMENT_COLORS = {
    "supportive": "#2EC4B6",
    "critical":   "#E63946",
    "suggestive": "#F4A261",
    "neutral":    "#8D99AE",
    "ambiguous":  "#9B5DE5",
}


class SHAPExplainer:
    """Token attribution using SHAP for transformer models."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        cache_dir: str = SHAP_CACHE_DIR,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._shap_explainer = None

    def _cache_key(self, sentence: str, label: str) -> str:
        return hashlib.md5(f"{sentence}:{label}".encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[dict]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        with open(self.cache_dir / f"{key}.json", "w") as f:
            json.dump(data, f)

    def _get_shap_values(self, sentence: str) -> Optional[dict]:
        """Compute SHAP values for a single sentence."""
        if self.model is None or self.tokenizer is None:
            return None
        try:
            import shap
            import torch

            if self._shap_explainer is None:
                # Use partition explainer (works with any model)
                def predict_fn(texts):
                    encodings = self.tokenizer(
                        list(texts),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128,
                    )
                    device = next(self.model.parameters()).device
                    encodings = {k: v.to(device) for k, v in encodings.items()}
                    with torch.no_grad():
                        outputs = self.model(**encodings)
                        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                    return probs

                self._shap_explainer = shap.Explainer(predict_fn, self.tokenizer)

            shap_values = self._shap_explainer([sentence], fixed_context=1)
            tokens = self.tokenizer.tokenize(sentence)
            # Average across all output classes for simplicity
            avg_values = shap_values.values[0].mean(axis=-1)[:len(tokens)]
            return {"tokens": tokens, "shap_values": avg_values.tolist()}

        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}")
            return None

    def explain(self, sentence: str, predicted_label: str) -> dict:
        """
        Get token attribution for a sentence.
        Returns dict with tokens and their importance scores.
        """
        cache_key = self._cache_key(sentence, predicted_label)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        shap_result = self._get_shap_values(sentence)
        if shap_result:
            result = {**shap_result, "label": predicted_label, "source": "shap"}
        else:
            # Fallback: simple keyword attribution
            result = self._keyword_attribution(sentence, predicted_label)

        self._save_cache(cache_key, result)
        return result

    @staticmethod
    def _keyword_attribution(sentence: str, label: str) -> dict:
        """Heuristic keyword-based attribution as SHAP fallback."""
        POSITIVE_WORDS = {"support", "welcome", "agree", "appreciate", "endorse", "commend", "favor", "good"}
        NEGATIVE_WORDS = {"oppose", "reject", "against", "critical", "concern", "harmful", "bad", "wrong", "reject"}
        SUGGEST_WORDS = {"suggest", "recommend", "propose", "amend", "revise", "consider", "should", "must"}

        tokens = sentence.split()
        scores = []
        for token in tokens:
            t_lower = token.lower().strip(".,;:!?()")
            if t_lower in POSITIVE_WORDS:
                scores.append(0.4 if label == "supportive" else -0.2)
            elif t_lower in NEGATIVE_WORDS:
                scores.append(-0.4 if label == "critical" else 0.1)
            elif t_lower in SUGGEST_WORDS:
                scores.append(0.3 if label == "suggestive" else 0.05)
            else:
                scores.append(0.0)

        return {"tokens": tokens, "shap_values": scores, "label": label, "source": "keyword_heuristic"}

    def to_html(self, explanation: dict, predicted_label: str) -> str:
        """Render token attributions as highlighted HTML."""
        tokens = explanation.get("tokens", [])
        values = explanation.get("shap_values", [])
        if not tokens:
            return "<span>No explanation available</span>"

        color = SENTIMENT_COLORS.get(predicted_label, "#8D99AE")
        html_parts = []

        max_abs = max(abs(v) for v in values) if values else 1.0
        if max_abs == 0:
            max_abs = 1.0

        for token, value in zip(tokens, values):
            norm = value / max_abs  # -1 to 1
            alpha = abs(norm) * 0.7 + 0.1
            if norm > 0.05:
                bg = f"rgba(46, 196, 182, {alpha})"  # teal for supporting evidence
            elif norm < -0.05:
                bg = f"rgba(230, 57, 70, {alpha})"    # red for opposing evidence
            else:
                bg = "transparent"

            clean_token = token.replace("##", "")  # Remove wordpiece markers
            html_parts.append(
                f'<span style="background:{bg};border-radius:3px;padding:1px 3px;margin:1px;" '
                f'title="SHAP: {value:.3f}">{clean_token}</span>'
            )

        return " ".join(html_parts)
