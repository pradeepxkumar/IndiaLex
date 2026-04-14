"""
IndiaLexABSA — Unit Tests: Dashboard Components
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from dashboard.components.sentiment_badge import sentiment_badge_html, sentiment_color, render_legend
from dashboard.components.kpi_card import kpi_card_html, mini_kpi_html
from dashboard.components.demo_data import get_demo_sentences, get_demo_docs, get_demo_heatmap_data


# ── Sentiment badge tests ─────────────────────────────────────

def test_badge_all_labels():
    for label in ["supportive", "critical", "suggestive", "neutral", "ambiguous"]:
        html = sentiment_badge_html(label)
        assert label.capitalize() in html or label in html.lower()


def test_badge_contains_color():
    html = sentiment_badge_html("critical")
    assert "#E63946" in html


def test_badge_large_size():
    html = sentiment_badge_html("supportive", size="lg")
    assert "0.85rem" in html


def test_sentiment_color():
    assert sentiment_color("supportive") == "#2EC4B6"
    assert sentiment_color("critical") == "#E63946"
    assert sentiment_color("suggestive") == "#F4A261"
    assert sentiment_color("neutral") == "#8D99AE"
    assert sentiment_color("ambiguous") == "#9B5DE5"


def test_render_legend():
    legend = render_legend()
    assert "Supportive" in legend
    assert "Critical" in legend


# ── KPI card tests ────────────────────────────────────────────

def test_kpi_card_contains_value():
    html = kpi_card_html("Total Comments", "1,234", sub="Across 5 submitters")
    assert "1,234" in html


def test_kpi_card_contains_label():
    html = kpi_card_html("Macro F1", "0.838")
    assert "Macro F1" in html


def test_mini_kpi_html():
    html = mini_kpi_html("F1", "0.84", "#2EC4B6")
    assert "0.84" in html
    assert "#2EC4B6" in html


# ── Demo data tests ───────────────────────────────────────────

def test_demo_sentences_count():
    sents = get_demo_sentences(100)
    assert len(sents) == 100


def test_demo_sentences_schema():
    sents = get_demo_sentences(10)
    for s in sents:
        assert "sent_id" in s
        assert "label" in s
        assert s["label"] in ["supportive", "critical", "suggestive", "neutral", "ambiguous"]
        assert "sentence" in s or "text" in s
        assert "clause_id" in s
        assert "submitter" in s


def test_demo_docs():
    docs = get_demo_docs()
    assert len(docs) == 3
    for d in docs:
        assert "doc_id" in d
        assert "submitter" in d
        assert "category" in d


def test_demo_heatmap_data():
    data = get_demo_heatmap_data()
    assert "clauses" in data
    assert "matrix" in data
    assert len(data["clauses"]) > 0
    for cid, counts in data["matrix"].items():
        assert isinstance(counts, dict)
        assert "supportive" in counts


def test_demo_sentence_confidence_range():
    sents = get_demo_sentences(50)
    for s in sents:
        assert 0.0 <= s.get("confidence", 0.5) <= 1.0


def test_demo_all_labels_present():
    sents = get_demo_sentences(287)
    labels_present = set(s.get("label") for s in sents)
    # With 287 sentences, all 5 labels should appear
    assert len(labels_present) == 5
