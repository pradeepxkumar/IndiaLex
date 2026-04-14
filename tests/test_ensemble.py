"""
IndiaLexABSA — Unit Tests: Ensemble Predictor
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from models.ensemble import EnsemblePredictor

LABELS = ["supportive", "critical", "suggestive", "neutral", "ambiguous"]

@pytest.fixture
def predictor():
    return EnsemblePredictor(demo_mode=True)


def test_predict_one_returns_label(predictor):
    result = predictor.predict_one("We strongly oppose Section 3.")
    assert "label" in result
    assert result["label"] in LABELS


def test_predict_one_confidence_range(predictor):
    result = predictor.predict_one("We support this provision.")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_one_probabilities_sum(predictor):
    result = predictor.predict_one("This clause requires data sharing.")
    probs = result.get("probabilities", {})
    assert abs(sum(probs.values()) - 1.0) < 0.05


def test_predict_batch(predictor):
    sentences = [
        "We oppose this clause strongly.",
        "We welcome the provisions.",
        "We suggest amending the definition.",
    ]
    results = predictor.predict_batch(sentences)
    assert len(results) == 3
    for r in results:
        assert r["label"] in LABELS


def test_critical_keywords(predictor):
    result = predictor.predict_one("We strongly oppose and reject this mandatory provision.")
    assert result["label"] == "critical"


def test_supportive_keywords(predictor):
    result = predictor.predict_one("We strongly support and welcome this framework.")
    assert result["label"] == "supportive"


def test_suggestive_keywords(predictor):
    result = predictor.predict_one("We strongly recommend and suggest amending this clause.")
    assert result["label"] == "suggestive"


def test_stats_tracking(predictor):
    for i in range(5):
        predictor.predict_one(f"Test sentence {i}.")
    stats = predictor.get_stats()
    assert stats["total"] == 5


def test_demo_mode_no_model_load(predictor):
    assert predictor._inlegalbert is None
    assert predictor._deberta is None
    result = predictor.predict_one("Sample sentence.")
    assert result["source"] == "demo"
