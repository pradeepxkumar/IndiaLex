"""
IndiaLexABSA — Clause Card Component
"""
from __future__ import annotations
from dashboard.components.sentiment_badge import sentiment_badge_html, sentiment_color


def clause_card_html(
    clause_id: str,
    title: str,
    text: str = "",
    dominant_label: str = "neutral",
    comment_count: int = 0,
    complexity: int = 5,
    show_expand: bool = True,
    max_text_len: int = 200,
) -> str:
    color = sentiment_color(dominant_label)
    badge = sentiment_badge_html(dominant_label)
    truncated = text[:max_text_len] + "..." if len(text) > max_text_len else text
    complexity_dots = "".join(
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{"#F4A261" if i < complexity else "#EDF2F7"};margin-right:2px;"></span>'
        for i in range(10)
    )

    return f"""
    <div style="background:white;border:1px solid #E2E8F0;border-radius:12px;
         padding:1rem 1.25rem;border-left:4px solid {color};
         box-shadow:0 1px 3px rgba(0,0,0,0.05);margin-bottom:0.5rem;
         transition:box-shadow 0.2s;"
         onmouseover="this.style.boxShadow='0 4px 12px rgba(0,0,0,0.10)'"
         onmouseout="this.style.boxShadow='0 1px 3px rgba(0,0,0,0.05)'">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.5rem;">
            <div>
                <span style="font-size:0.72rem;font-weight:700;color:#A0AEC0;letter-spacing:0.5px;">
                    {clause_id}
                </span>
                <div style="font-weight:600;font-size:0.9rem;color:#2D3748;margin-top:1px;">{title}</div>
            </div>
            <div style="text-align:right;">
                {badge}
                <div style="font-size:0.7rem;color:#A0AEC0;margin-top:3px;">{comment_count} comments</div>
            </div>
        </div>
        {f'<div style="font-size:0.82rem;color:#718096;line-height:1.6;margin-bottom:0.5rem;">{truncated}</div>' if truncated else ''}
        <div style="display:flex;align-items:center;gap:0.5rem;">
            <span style="font-size:0.7rem;color:#A0AEC0;">Complexity:</span>
            {complexity_dots}
            <span style="font-size:0.7rem;color:#A0AEC0;">{complexity}/10</span>
        </div>
    </div>
    """
