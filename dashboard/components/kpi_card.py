"""
IndiaLexABSA — KPI Card Component
"""
from __future__ import annotations


def kpi_card_html(
    label: str,
    value: str,
    sub: str = "",
    color: str = "#2EC4B6",
    icon: str = "",
    delta: str = "",
    delta_positive: bool = True,
) -> str:
    """Render a KPI metric card as HTML."""
    delta_html = ""
    if delta:
        delta_color = "#276749" if delta_positive else "#C53030"
        delta_arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div style="font-size:0.75rem;color:{delta_color};font-weight:600;margin-top:2px;">{delta_arrow} {delta}</div>'

    return f"""
    <div style="background:white;border:1px solid #E2E8F0;border-radius:14px;
         padding:1.25rem 1.5rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);
         transition:box-shadow 0.2s;cursor:default;"
         onmouseover="this.style.boxShadow='0 4px 16px rgba(0,0,0,0.10)'"
         onmouseout="this.style.boxShadow='0 1px 4px rgba(0,0,0,0.06)'">
        <div style="font-size:0.72rem;font-weight:600;color:#718096;
             text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.4rem;">
            {icon + ' ' if icon else ''}{label}
        </div>
        <div style="font-size:1.9rem;font-weight:700;color:{color};line-height:1.15;">
            {value}
        </div>
        {delta_html}
        <div style="font-size:0.75rem;color:#A0AEC0;margin-top:0.3rem;">{sub}</div>
    </div>
    """


def mini_kpi_html(label: str, value: str, color: str = "#2EC4B6") -> str:
    """Compact inline KPI for use inside panels."""
    return f"""
    <div style="display:inline-block;background:{color}10;border:1px solid {color}30;
         border-radius:8px;padding:0.4rem 0.8rem;margin:0.2rem;">
        <span style="font-size:0.72rem;color:#718096;">{label}: </span>
        <span style="font-size:0.85rem;font-weight:700;color:{color};">{value}</span>
    </div>
    """
