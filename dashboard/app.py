"""
IndiaLexABSA — Home Page (app.py)
Run:  streamlit run dashboard/app.py   (from project root)
  or: streamlit run app.py             (from dashboard/)
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.components.shared import page_setup

p = page_setup("IndiaLexABSA", "⚖️", "Home")

BG, CARD, BD, TX, MU = p["bg"], p["card"], p["bd"], p["tx"], p["mu"]

import streamlit as st

# ── Hero ───────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:2rem 0 1.8rem;">
    <h1 style="font-size:2.9rem;font-weight:900;color:{TX};
               margin:0;letter-spacing:-2px;line-height:1.1;">
        ⚖️ IndiaLex
    </h1>
    <p style="color:{MU};opacity:0.6;font-size:1rem;margin:0.75rem 0 0;line-height:1.75;">
        AI-Powered Aspect-Based Sentiment Analysis for<br>
        Indian Regulatory Stakeholder Comments
    </p>
    <div style="display:inline-flex;gap:0.5rem;margin-top:1rem;
                flex-wrap:wrap;justify-content:center;">
        <span style="background:rgba(46,196,182,0.12);color:#2EC4B6;
              border:1px solid rgba(46,196,182,0.3);border-radius:20px;
              padding:4px 14px;font-size:0.78rem;font-weight:600;">InLegalBERT</span>
        <span style="background:rgba(99,102,241,0.12);color:#6366F1;
              border:1px solid rgba(99,102,241,0.3);border-radius:20px;
              padding:4px 14px;font-size:0.78rem;font-weight:600;">DeBERTa-v3</span>
        <span style="background:rgba(245,158,11,0.12);color:#F59E0B;
              border:1px solid rgba(245,158,11,0.3);border-radius:20px;
              padding:4px 14px;font-size:0.78rem;font-weight:600;">SBERT</span>
        <span style="background:rgba(236,72,153,0.12);color:#EC4899;
              border:1px solid rgba(236,72,153,0.3);border-radius:20px;
              padding:4px 14px;font-size:0.78rem;font-weight:600;">IndicTrans2</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Feature cards ─────────────────────────────────────────────
FEATURES = [
    ("🤖", "Ensemble AI",   "InLegalBERT + DeBERTa-v3<br>soft-vote ensemble",  "#2EC4B6"),
    ("⚖️",  "47 Clauses",   "Full Digital Competition<br>Bill clause analysis",   "#6366F1"),
    ("🌐", "Hindi Support", "IndicTrans2 multilingual<br>translation pipeline",   "#F59E0B"),
    ("📄", "PDF Reports",   "Professional government-<br>grade PDF export",       "#EC4899"),
]
for col, (ico, title, desc, accent) in zip(st.columns(4, gap="medium"), FEATURES):
    col.markdown(f"""
    <div style="background:{CARD};border:1.5px solid {BD};
         border-top:4px solid {accent};border-radius:14px;
         padding:1.2rem 1rem;text-align:center;
         box-shadow:0 1px 5px rgba(0,0,0,0.1);">
        <div style="font-size:1.8rem;margin-bottom:0.4rem;">{ico}</div>
        <div style="font-weight:700;color:{TX};font-size:0.87rem;margin-bottom:0.2rem;">{title}</div>
        <div style="color:{MU};opacity:0.55;font-size:0.71rem;line-height:1.5;">{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────
STATS = [("~5,000","Labeled Triples"),("47","DCB Clauses"),("5","Sentiment Classes"),("84.7%","Ensemble F1")]
for col, (num, label) in zip(st.columns(4, gap="medium"), STATS):
    col.markdown(f"""
    <div style="text-align:center;padding:0.9rem;background:{CARD};
         border:1.5px solid {BD};border-radius:12px;
         box-shadow:0 1px 4px rgba(0,0,0,0.08);">
        <div style="font-size:1.6rem;font-weight:800;color:#2EC4B6;line-height:1;">{num}</div>
        <div style="font-size:0.7rem;color:{MU};opacity:0.55;margin-top:4px;font-weight:500;">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)
st.markdown(
    f'<p style="text-align:center;font-size:0.74rem;color:{MU};opacity:0.6;line-height:1.7;margin:0;">'
    f'Use the sidebar to navigate. Start with <b>📤 Upload &amp; Process</b> to analyse PDFs, '
    f'or enable <b>Demo Mode</b> in the sidebar for instant sample results.</p>',
    unsafe_allow_html=True,
)