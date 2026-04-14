"""
IndiaLexABSA — Baseline Models
================================
Three baselines for comparison against fine-tuned transformers:
  1. VADER (rule-based sentiment, adapted to 5-class)
  2. TF-IDF + Logistic Regression
  3. TextBlob polarity bucketing

All share a consistent predict() interface returning:
  {"label": str, "confidence": float, "scores": dict}
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


LABELS = ["supportive", "critical", "suggestive", "neutral", "ambiguous"]


def _vader_to_5class(compound: float) -> str:
    """Map VADER compound score to 5-class scheme."""
    if compound >= 0.35:
        return "supportive"
    elif compound <= -0.35:
        return "critical"
    elif -0.15 <= compound <= 0.15:
        return "neutral"
    elif compound > 0.15:
        return "suggestive"
    else:
        return "ambiguous"


class VADERBaseline:
    """VADER sentiment analysis adapted to 5-class scheme."""

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            try:
                import nltk
                nltk.download("vader_lexicon", quiet=True)
                from nltk.sentiment import SentimentIntensityAnalyzer
                self.analyzer = SentimentIntensityAnalyzer()
            except Exception as exc:
                logger.error(f"VADER unavailable: {exc}")
                self.analyzer = None

    def predict(self, sentences: list[str]) -> list[dict]:
        results = []
        for sent in sentences:
            if self.analyzer is None:
                results.append({"label": "neutral", "confidence": 0.2, "scores": {}})
                continue
            scores = self.analyzer.polarity_scores(sent)
            label = _vader_to_5class(scores["compound"])
            confidence = abs(scores["compound"])
            results.append({"label": label, "confidence": confidence, "scores": scores})
        return results

    def predict_one(self, sentence: str) -> dict:
        return self.predict([sentence])[0]


class TFIDFLogisticBaseline:
    """TF-IDF + Logistic Regression classifier."""

    def __init__(self, max_features: int = 50_000, C: float = 1.0):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, sentences: list[str], labels: list[str]) -> "TFIDFLogisticBaseline":
        y = self.label_encoder.fit_transform(labels)
        self.pipeline.fit(sentences, y)
        self.is_fitted = True
        logger.info("TF-IDF + LR baseline fitted")
        return self

    def predict(self, sentences: list[str]) -> list[dict]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        proba = self.pipeline.predict_proba(sentences)
        labels_encoded = self.pipeline.predict(sentences)
        results = []
        for i, sent in enumerate(sentences):
            label = self.label_encoder.inverse_transform([labels_encoded[i]])[0]
            confidence = float(proba[i].max())
            scores = {
                self.label_encoder.inverse_transform([j])[0]: float(proba[i][j])
                for j in range(len(proba[i]))
            }
            results.append({"label": label, "confidence": confidence, "scores": scores})
        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "encoder": self.label_encoder}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pipeline = data["pipeline"]
        self.label_encoder = data["encoder"]
        self.is_fitted = True


class TextBlobBaseline:
    """TextBlob polarity-based 5-class classifier."""

    def predict(self, sentences: list[str]) -> list[dict]:
        try:
            from textblob import TextBlob
        except ImportError:
            return [{"label": "neutral", "confidence": 0.2, "scores": {}} for _ in sentences]

        results = []
        for sent in sentences:
            blob = TextBlob(sent)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            if polarity >= 0.3:
                label = "supportive"
            elif polarity <= -0.3:
                label = "critical"
            elif subjectivity > 0.6 and -0.1 <= polarity <= 0.3:
                label = "suggestive"
            elif abs(polarity) < 0.1 and subjectivity < 0.3:
                label = "neutral"
            else:
                label = "ambiguous"

            results.append({
                "label": label,
                "confidence": abs(polarity),
                "scores": {"polarity": polarity, "subjectivity": subjectivity},
            })
        return results
