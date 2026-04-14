"""IndiaLexABSA — Page 2: Executive Overview"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.components.shared import page_setup
p = page_setup("Overview — IndiaLexABSA", "📊", "02_overview")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dashboard.components.demo_data import get_demo_sentences

BG, CARD, BD, TX, MU = p["bg"], p["card"], p["bd"], p["tx"], p["mu"]

COLORS = {
    "supportive": "#2EC4B6", "critical": "#E63946",
    "suggestive": "#F59E0B", "neutral":  "#94A3B8", "ambiguous": "#8B5CF6",
}

st.markdown(f"""
<div style="border-bottom:1px solid {BD};padding-bottom:1rem;margin-bottom:1.4rem;">
    <h1 style="font-size:1.7rem;font-weight:800;color:{TX};margin:0;letter-spacing:-0.5px;">
        📊 Executive Overview
    </h1>
    <p style="color:{MU};opacity:0.55;margin:0.3rem 0 0;font-size:0.88rem;">
        Aggregated sentiment analysis across all uploaded stakeholder submissions.
    </p>
</div>""", unsafe_allow_html=True)

if not st.session_state.processing_done:
    st.info("⚠️ No data yet. Go to **📤 Upload & Process** and run the pipeline first, or enable **Demo Mode**.")
    st.stop()

sentences = st.session_state.all_sentences or get_demo_sentences()

# ── KPI cards ─────────────────────────────────────────────────
from collections import Counter
labels   = [s.get("label", "neutral") for s in sentences]
counts   = Counter(labels)
total    = len(sentences)
dominant = counts.most_common(1)[0][0] if counts else "neutral"
dom_pct  = int(counts[dominant] / total * 100) if total else 0
conf_avg = int(sum(s.get("confidence", 0.7) for s in sentences) / max(total, 1) * 100)
docs_n   = len(st.session_state.uploaded_docs)

kpi_data = [
    ("📄", str(docs_n),       "Submissions",        "#2EC4B6"),
    ("📝", str(total),        "Sentences Analysed", "#6366F1"),
    ("🎯", f"{dom_pct}%", f"Dominant: {dominant.title()}", COLORS.get(dominant, "#2EC4B6")),
    ("🎯", f"{conf_avg}%",   "Avg Confidence",     "#F59E0B"),
]

k1, k2, k3, k4 = st.columns(4, gap="medium")
for col, (icon, value, label, accent) in zip([k1, k2, k3, k4], kpi_data):
    col.markdown(f"""
    <div style="background:{CARD};border:1.5px solid {BD};border-top:4px solid {accent};
         border-radius:14px;padding:1rem;text-align:center;">
        <div style="font-size:1.6rem;font-weight:800;color:{accent};">{value}</div>
        <div style="font-size:0.72rem;color:{MU};opacity:0.55;margin-top:3px;">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # Donut chart
    st.markdown(f"<div style='font-weight:700;font-size:0.95rem;color:{TX};margin-bottom:0.5rem;'>Sentiment Distribution</div>", unsafe_allow_html=True)
    labels_list  = list(counts.keys())
    values_list  = list(counts.values())
    colors_list  = [COLORS.get(l, "#94A3B8") for l in labels_list]
    fig_donut = go.Figure(go.Pie(
        labels=[l.title() for l in labels_list],
        values=values_list,
        hole=0.6,
        marker_colors=colors_list,
        textinfo="percent+label",
        textfont_size=12,
    ))
    fig_donut.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        annotations=[dict(text=f"<b>{dom_pct}%<br>{dominant}</b>", x=0.5, y=0.5,
                          font_size=14, showarrow=False, font_color=COLORS.get(dominant, "#2EC4B6"))],
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_right:
    # Clause bar chart
    st.markdown(f"<div style='font-weight:700;font-size:0.95rem;color:{TX};margin-bottom:0.5rem;'>Top Clauses by Comment Volume</div>", unsafe_allow_html=True)
    clause_counts: Counter = Counter(s.get("clause_title", "Unknown") for s in sentences)
    top_clauses = clause_counts.most_common(8)
    if top_clauses:
        clause_names = [c[:30] + "…" if len(c) > 30 else c for c, _ in top_clauses]
        clause_vals  = [v for _, v in top_clauses]
        fig_bar = go.Figure(go.Bar(
            x=clause_vals, y=clause_names, orientation="h",
            marker_color="#2EC4B6",
            marker_line_width=0,
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
            xaxis=dict(showgrid=False),
            yaxis=dict(autorange="reversed"),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── Executive brief (demo) ────────────────────────────────────
with st.expander("🤖 AI Executive Brief", expanded=True):
    st.markdown(f"""
    <div style="color:{TX};font-size:0.88rem;line-height:1.75;padding:0.2rem 0;">
        <p>Stakeholder submissions on the Digital Competition Bill 2024 reveal a predominantly
        <b>critical sentiment</b> toward provisions on data sharing obligations (Clause 12–15) and
        market definition criteria (Clause 3). Respondents from the technology sector express concerns
        about implementation timelines and definitional ambiguity.</p>
        <p>Supportive sentiment is concentrated around consumer protection provisions (Clause 22–26),
        with industry associations endorsing the interoperability mandate while requesting phased rollout.
        Small business representatives show the highest proportion of suggestive responses, proposing
        carve-outs for MSMEs.</p>
        <p><b>Key policy recommendation:</b> Prioritise clarification of "significant digital enterprise"
        thresholds in Clause 3(1)(a) and establish a dedicated implementation task force to address
        the 37% of critical submissions citing regulatory uncertainty as a primary concern.</p>
    </div>
    """, unsafe_allow_html=True)
