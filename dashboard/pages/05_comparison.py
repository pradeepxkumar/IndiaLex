"""IndiaLexABSA — Page 5: Compare Documents"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.components.shared import page_setup
p = page_setup("Compare — IndiaLexABSA", "⚖️", "05_comparison")

import streamlit as st
import plotly.graph_objects as go
from collections import Counter, defaultdict
from dashboard.components.demo_data import get_demo_sentences, get_demo_docs

BG, CARD, BD, TX, MU = p["bg"], p["card"], p["bd"], p["tx"], p["mu"]

COLORS = {
    "supportive": "#2EC4B6", "critical": "#E63946",
    "suggestive": "#F59E0B", "neutral":  "#94A3B8", "ambiguous": "#8B5CF6",
}
EMOJI = {
    "supportive": "🟢", "critical": "🔴",
    "suggestive": "🟡", "neutral": "⚫", "ambiguous": "🟣",
}

st.markdown(f"""
<div style="border-bottom:1px solid {BD};padding-bottom:1rem;margin-bottom:1.4rem;">
    <h1 style="font-size:1.7rem;font-weight:800;color:{TX};margin:0;letter-spacing:-0.5px;">
        ⚖️ Compare Documents
    </h1>
    <p style="color:{MU};opacity:0.55;margin:0.3rem 0 0;font-size:0.88rem;">
        Side-by-side sentiment comparison across multiple stakeholder submissions.
    </p>
</div>""", unsafe_allow_html=True)

if not st.session_state.processing_done:
    st.info("⚠️ No data yet. Go to **📤 Upload & Process** first, or enable **Demo Mode**.")
    st.stop()

sentences = st.session_state.all_sentences or get_demo_sentences()
docs      = st.session_state.uploaded_docs  or get_demo_docs()

# Group sentences by submitter/doc
doc_names = sorted({s.get("submitter", "Unknown") for s in sentences})

# Select docs to compare
selected_docs = st.multiselect(
    "Select documents to compare (2–5):",
    options=doc_names,
    default=doc_names[:min(3, len(doc_names))],
)
if len(selected_docs) < 2:
    st.warning("Please select at least 2 documents to compare.")
    st.stop()

# Build comparison matrix: clause → doc → dominant label
clause_titles = sorted({s.get("clause_title","Unknown") for s in sentences})
matrix: dict = defaultdict(dict)
for s in sentences:
    sub    = s.get("submitter","Unknown")
    clause = s.get("clause_title","Unknown")
    if sub in selected_docs:
        if clause not in matrix[sub]:
            matrix[sub][clause] = []
        matrix[sub][clause].append(s.get("label","neutral"))

st.markdown(f"<div style='font-weight:700;font-size:0.95rem;color:{TX};margin-bottom:0.7rem;'>Sentiment Matrix</div>", unsafe_allow_html=True)

# Header row
cols = st.columns([2] + [1]*len(selected_docs))
cols[0].markdown(f"<div style='font-size:0.75rem;font-weight:700;color:{MU};opacity:0.55;padding:0.3rem 0;'>Clause</div>", unsafe_allow_html=True)
for i, doc in enumerate(selected_docs):
    cols[i+1].markdown(f"<div style='font-size:0.73rem;font-weight:700;color:{TX};padding:0.3rem 0;'>{doc[:16]}</div>", unsafe_allow_html=True)

st.markdown(f"<hr style='border-color:{BD};margin:0.3rem 0 0.5rem;'>", unsafe_allow_html=True)

# Data rows
for clause in clause_titles:
    # Compute dominant labels
    dom_labels = {}
    for doc in selected_docs:
        lbls = matrix.get(doc, {}).get(clause, [])
        if lbls:
            dom = Counter(lbls).most_common(1)[0][0]
            dom_labels[doc] = dom
        else:
            dom_labels[doc] = None

    # Skip clause if no doc has data
    if not any(dom_labels.values()):
        continue

    # Consensus check
    active = [v for v in dom_labels.values() if v]
    consensus = len(set(active)) == 1 if active else True

    row_cols = st.columns([2] + [1]*len(selected_docs))
    row_cols[0].markdown(
        f'<div style="font-size:0.75rem;color:{TX};padding:0.25rem 0;">{clause[:28]}{"…" if len(clause)>28 else ""}'
        f'{"&nbsp;⚠️" if not consensus else ""}</div>',
        unsafe_allow_html=True,
    )
    for i, doc in enumerate(selected_docs):
        dom = dom_labels[doc]
        if dom:
            emoji = EMOJI.get(dom,"⚫")
            color = COLORS.get(dom,"#94A3B8")
            row_cols[i+1].markdown(
                f'<div style="font-size:0.8rem;text-align:center;padding:0.25rem 0;'
                f'color:{color};font-weight:600;">{emoji} {dom[:4]}</div>',
                unsafe_allow_html=True,
            )
        else:
            row_cols[i+1].markdown(f'<div style="font-size:0.8rem;text-align:center;color:{MU};opacity:0.55;padding:0.25rem 0;">—</div>', unsafe_allow_html=True)

st.markdown(f"<hr style='border-color:{BD};margin:1rem 0;'>", unsafe_allow_html=True)

# Consensus meter
agree_count = 0
total_clauses = 0
for clause in clause_titles:
    active = []
    for doc in selected_docs:
        lbls = matrix.get(doc, {}).get(clause, [])
        if lbls:
            active.append(Counter(lbls).most_common(1)[0][0])
    if len(active) >= 2:
        total_clauses += 1
        if len(set(active)) == 1:
            agree_count += 1

if total_clauses > 0:
    pct = int(agree_count / total_clauses * 100)
    st.markdown(f"""
    <div style="background:{CARD};border:1.5px solid {BD};border-radius:12px;padding:1rem 1.2rem;">
        <div style="font-weight:700;font-size:0.9rem;color:{TX};margin-bottom:0.5rem;">📊 Consensus Meter</div>
        <div style="font-size:0.8rem;color:{MU};opacity:0.55;margin-bottom:0.5rem;">
            {agree_count} of {total_clauses} clauses show full agreement across selected documents
        </div>
        <div style="background:{BD};border-radius:6px;height:10px;overflow:hidden;">
            <div style="width:{pct}%;height:10px;background:#2EC4B6;border-radius:6px;
                        transition:width 0.5s;"></div>
        </div>
        <div style="font-size:0.78rem;color:#2EC4B6;font-weight:700;margin-top:4px;">{pct}% agreement</div>
    </div>""", unsafe_allow_html=True)
