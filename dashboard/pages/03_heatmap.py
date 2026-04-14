"""IndiaLexABSA — Page 3: Clause Heatmap"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.components.shared import page_setup
p = page_setup("Heatmap — IndiaLexABSA", "🗺️", "03_heatmap")

import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict
from dashboard.components.demo_data import get_demo_sentences

BG, CARD, BD, TX, MU = p["bg"], p["card"], p["bd"], p["tx"], p["mu"]

SENTIMENTS = ["supportive", "critical", "suggestive", "neutral", "ambiguous"]
COLORS = {
    "supportive": "#2EC4B6", "critical": "#E63946",
    "suggestive": "#F59E0B", "neutral":  "#94A3B8", "ambiguous": "#8B5CF6",
}

st.markdown(f"""
<div style="border-bottom:1px solid {BD};padding-bottom:1rem;margin-bottom:1.4rem;">
    <h1 style="font-size:1.7rem;font-weight:800;color:{TX};margin:0;letter-spacing:-0.5px;">
        🗺️ Clause Heatmap
    </h1>
    <p style="color:{MU};opacity:0.55;margin:0.3rem 0 0;font-size:0.88rem;">
        Interactive heatmap showing sentiment distribution across all 47 DCB clauses.
    </p>
</div>""", unsafe_allow_html=True)

if not st.session_state.processing_done:
    st.info("⚠️ No data yet. Go to **📤 Upload & Process** first, or enable **Demo Mode**.")
    st.stop()

sentences = st.session_state.all_sentences or get_demo_sentences()

# Filters
f1, f2, _ = st.columns([2, 2, 3])
with f1:
    conf_threshold = st.slider("Min confidence", 0.0, 1.0, 0.3, 0.05)
with f2:
    lang_filter = st.selectbox("Language", ["All", "English", "Hindi"])

filtered = [s for s in sentences if (s.get("confidence") or 0.7) >= conf_threshold]
if lang_filter != "All":
    filtered = [s for s in filtered if s.get("language", "en") == ("hi" if lang_filter == "Hindi" else "en")]

# Build matrix
clause_titles = sorted({s.get("clause_title", "Unknown") for s in filtered})[:15]
matrix = [[0]*len(SENTIMENTS) for _ in clause_titles]
for s in filtered:
    ct = s.get("clause_title", "Unknown")
    lb = s.get("label", "neutral")
    if ct in clause_titles and lb in SENTIMENTS:
        matrix[clause_titles.index(ct)][SENTIMENTS.index(lb)] += 1

# Heatmap
z_vals = matrix
fig = go.Figure(go.Heatmap(
    z=z_vals,
    x=[s.title() for s in SENTIMENTS],
    y=[c[:28]+"…" if len(c)>28 else c for c in clause_titles],
    colorscale=[[0,"rgba(30,41,59,0.3)"], [0.5,"#2EC4B6"], [1,"#0D9488"]],
    showscale=True,
    hovertemplate="<b>%{y}</b><br>%{x}: %{z} sentences<extra></extra>",
))
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=20, b=20, l=20, r=20),
    height=420,
    font=dict(family="Inter", size=11),
    xaxis=dict(side="top"),
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"<div style='font-size:0.76rem;color:{MU};opacity:0.55;text-align:center;margin-top:-0.5rem;'>Showing {len(filtered)} sentences across {len(clause_titles)} clauses</div>", unsafe_allow_html=True)
st.markdown(f"<hr style='border-color:{BD};margin:1rem 0;'>", unsafe_allow_html=True)

# Sentence panel when filtering
st.markdown(f"<div style='font-weight:700;font-size:0.95rem;color:{TX};margin-bottom:0.6rem;'>🔍 Filtered Sentences</div>", unsafe_allow_html=True)
selected_sentiment = st.selectbox("Filter by sentiment", ["All"] + [s.title() for s in SENTIMENTS])

show = filtered
if selected_sentiment != "All":
    show = [s for s in filtered if s.get("label","").lower() == selected_sentiment.lower()]

for s in show[:8]:
    label  = s.get("label", "neutral")
    color  = COLORS.get(label, "#94A3B8")
    text   = s.get("sentence", s.get("text", ""))
    clause = s.get("clause_title", "Unknown")
    conf   = int((s.get("confidence") or 0.7) * 100)
    st.markdown(f"""
    <div class="s-card" style="border-left-color:{color};">
        <div class="s-text">{text[:220]}{'…' if len(text)>220 else ''}</div>
        <div class="s-meta">
            <span class="s-badge" style="background:{color}22;color:{color};">{label.title()}</span>
            <span class="s-clause">⚖️ {clause[:30]}</span>
            <span class="s-sub" style="margin-left:auto;">{conf}% conf.</span>
        </div>
    </div>""", unsafe_allow_html=True)
if len(show) > 8:
    st.markdown(f"<div style='font-size:0.75rem;color:{MU};opacity:0.55;text-align:center;'>…and {len(show)-8} more sentences</div>", unsafe_allow_html=True)
