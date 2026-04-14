"""
IndiaLexABSA — Dashboard Components: Sentiment Badge
"""
from __future__ import annotations

COLORS = {
    "supportive": ("#2EC4B6", "#E6FAF9", "Supportive"),
    "critical":   ("#E63946", "#FEE8EA", "Critical"),
    "suggestive": ("#F4A261", "#FEF3E2", "Suggestive"),
    "neutral":    ("#8D99AE", "#F0F1F3", "Neutral"),
    "ambiguous":  ("#9B5DE5", "#F3ECFD", "Ambiguous"),
}


def sentiment_badge_html(label: str, size: str = "sm") -> str:
    """Return an HTML sentiment pill badge."""
    color, bg, display = COLORS.get(label, ("#8D99AE", "#F0F1F3", label.capitalize()))
    pad = "2px 10px" if size == "sm" else "4px 14px"
    font = "0.72rem" if size == "sm" else "0.85rem"
    return (
        f'<span style="background:{bg};color:{color};border:1px solid {color}33;'
        f'border-radius:20px;padding:{pad};font-size:{font};font-weight:600;'
        f'white-space:nowrap;display:inline-block;">'
        f'● {display}</span>'
    )


def sentiment_color(label: str) -> str:
    return COLORS.get(label, ("#8D99AE", "#F0F1F3", ""))[0]


def render_legend() -> str:
    """HTML color legend for all 5 sentiment classes."""
    badges = " ".join(sentiment_badge_html(l) for l in COLORS)
    return f'<div style="display:flex;gap:0.5rem;flex-wrap:wrap;align-items:center;">{badges}</div>'
