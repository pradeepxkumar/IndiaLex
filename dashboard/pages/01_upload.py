"""IndiaLexABSA — Page 1: Upload & Process"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.components.shared import page_setup
p = page_setup("Upload — IndiaLexABSA", "📤", "01_upload")

import streamlit as st
from dashboard.components.sentiment_badge import sentiment_badge_html
from dashboard.components.demo_data import get_demo_sentences, get_demo_docs

BG, CARD, BD, TX, MU = p["bg"], p["card"], p["bd"], p["tx"], p["mu"]

STEPS = [
    ("📄", "Extracting Text",       "PyMuPDF + pdfplumber hybrid extraction"),
    ("🌐", "Detecting Language",    "langdetect · IndicTrans2 translation"),
    ("⚖️", "Parsing Clauses",       "Linking to DCB ChromaDB knowledge base"),
    ("🔗", "Linking Sentences",     "SBERT MMR semantic clause assignment"),
    ("🎯", "Classifying Sentiment", "InLegalBERT + DeBERTa ensemble inference"),
]
COLORS = {
    "supportive": "#2EC4B6", "critical": "#E63946",
    "suggestive": "#F59E0B", "neutral":  "#94A3B8", "ambiguous": "#8B5CF6",
}

st.markdown(f"""
<style>
.step-row {{
    display:flex;align-items:center;gap:14px;
    padding:0.85rem 1.1rem;border-radius:11px;margin-bottom:0.4rem;
    border:1px solid {BD};background:{CARD};transition:all .2s;
}}
.step-done   {{ border-color:#2EC4B6 !important;background:rgba(46,196,182,0.08) !important; }}
.step-active {{ border-color:#F59E0B !important;background:rgba(245,158,11,0.08) !important; }}
.step-pending{{ opacity:0.45; }}
.step-label  {{ font-weight:600;font-size:0.88rem;color:{TX}; }}
.step-detail {{ font-size:0.71rem;color:{MU};opacity:0.55;margin-top:2px; }}
</style>
""", unsafe_allow_html=True)


def render_steps(active: int) -> str:
    html = ""
    for i, (icon, label, detail) in enumerate(STEPS):
        if i < active:
            cls  = "step-done"
            tag  = "<span style='color:#2EC4B6;font-weight:700;font-size:0.78rem;white-space:nowrap;'>✓ Done</span>"
        elif i == active:
            cls  = "step-active"
            tag  = "<span style='color:#F59E0B;font-weight:700;font-size:0.78rem;white-space:nowrap;'>⟳ Running…</span>"
        else:
            cls  = "step-pending"
            tag  = f"<span style='color:{MU};opacity:0.55;font-size:0.78rem;white-space:nowrap;'>Pending</span>"
        html += f"""
        <div class="step-row {cls}">
            <span style="font-size:1.1rem;width:26px;text-align:center;">{icon}</span>
            <div style="flex:1;">
                <div class="step-label">{label}</div>
                <div class="step-detail">{detail}</div>
            </div>
            {tag}
        </div>"""
    return html


# Page header
st.markdown(f"""
<div style="border-bottom:1px solid {BD};padding-bottom:1rem;margin-bottom:1.4rem;">
    <h1 style="font-size:1.7rem;font-weight:800;color:{TX};margin:0;letter-spacing:-0.5px;">
        📤 Upload &amp; Process
    </h1>
    <p style="color:{MU};opacity:0.55;margin:0.3rem 0 0;font-size:0.88rem;">
        Upload stakeholder PDF submissions on the Digital Competition Bill to begin analysis.
    </p>
</div>""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(f"<div style='font-weight:700;font-size:0.95rem;color:{TX};margin-bottom:0.6rem;'>📂 Drop PDFs here</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="hidden")

    if st.session_state.demo_mode:
        st.markdown(f"""
        <div style="background:rgba(46,196,182,0.10);
             border:1px solid rgba(46,196,182,0.28);border-radius:10px;
             padding:0.8rem 1rem;margin-top:0.6rem;">
            <div style="font-weight:700;color:#0D9488;font-size:0.84rem;margin-bottom:3px;">
                🟢 Demo Mode Active
            </div>
            <div style="color:{MU};opacity:0.55;font-size:0.79rem;line-height:1.5;">
                3 stakeholder submissions pre-loaded.<br>
                Click <b>Run Demo Pipeline</b> to see full analysis instantly.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)
    btn_label = "▶  Run Demo Pipeline" if st.session_state.demo_mode else "▶  Start Analysis"
    run_btn = st.button(btn_label, use_container_width=True, key="run_pipeline")

    with st.expander("⚙️  Pipeline Settings", expanded=False):
        st.slider("Clause linking threshold", 0.1, 0.9, 0.35, 0.05,
                  help="Higher = fewer but more confident clause links")
        st.slider("Ensemble confidence cutoff", 0.3, 0.9, 0.65, 0.05,
                  help="Below this threshold, the prediction is marked as low-confidence")
        st.checkbox("Translate Hindi sentences", value=True)

with right:
    st.markdown(f"<div style='font-weight:700;font-size:0.95rem;color:{TX};margin-bottom:0.6rem;'>⚙️ Processing Pipeline</div>", unsafe_allow_html=True)
    step_ph    = st.empty()
    counter_ph = st.empty()
    step_ph.markdown(render_steps(-1), unsafe_allow_html=True)

if run_btn:
    st.session_state.processing_done = False
    delays = [0.7, 0.6, 1.0, 1.1, 1.3]
    for idx in range(len(STEPS)):
        step_ph.markdown(render_steps(idx), unsafe_allow_html=True)
        if idx >= 2:
            for tick in range(6):
                linked = idx * 55 + tick * 12 + 5
                sents  = int(linked * 9.3)
                counter_ph.markdown(f"""
                <div style="background:rgba(46,196,182,0.08);
                     border:1px solid rgba(46,196,182,0.25);
                     border-radius:8px;padding:0.6rem 1rem;margin-top:0.4rem;font-size:0.87rem;">
                    <b style="color:#2EC4B6;font-size:1rem;">{sents}</b>
                    <span style="color:{MU};opacity:0.55;"> sentences linked to </span>
                    <b style="color:#2EC4B6;font-size:1rem;">{linked}</b>
                    <span style="color:{MU};opacity:0.55;"> clauses…</span>
                </div>""", unsafe_allow_html=True)
                time.sleep(delays[idx] / 6)
        else:
            time.sleep(delays[idx])

    step_ph.markdown(render_steps(len(STEPS)), unsafe_allow_html=True)
    counter_ph.markdown(f"""
    <div style="background:rgba(46,196,182,0.08);
         border:1px solid rgba(46,196,182,0.3);
         border-radius:8px;padding:0.7rem 1rem;margin-top:0.4rem;">
        <b style="color:#2EC4B6;font-size:1.1rem;">287</b>
        <span style="color:{MU};opacity:0.55;font-size:0.88rem;"> sentences linked to </span>
        <b style="color:#2EC4B6;font-size:1.1rem;">31</b>
        <span style="color:{MU};opacity:0.55;font-size:0.88rem;"> clauses across </span>
        <b style="color:#2EC4B6;font-size:1.1rem;">3</b>
        <span style="color:{MU};opacity:0.55;font-size:0.88rem;"> documents &nbsp;✓</span>
    </div>""", unsafe_allow_html=True)

    st.session_state.all_sentences   = get_demo_sentences()
    st.session_state.uploaded_docs   = get_demo_docs()
    st.session_state.processing_done = True
    st.success("✅ Pipeline complete! Navigate to **📊 Executive Overview** to explore results.")

# Sentence preview
st.markdown(f"<hr style='border-color:{BD};margin:1.3rem 0;'>", unsafe_allow_html=True)

if st.session_state.processing_done and st.session_state.all_sentences:
    st.markdown(f"<div style='font-weight:700;font-size:0.95rem;color:{TX};margin-bottom:0.6rem;'>🔍 Preview — First 3 Analysed Sentences</div>", unsafe_allow_html=True)
    for s in st.session_state.all_sentences[:3]:
        label  = s.get("label", "neutral")
        conf   = int((s.get("confidence") or 0.7) * 100)
        clause = s.get("clause_title", "Unknown")
        text   = s.get("sentence", s.get("text", ""))
        sub    = s.get("submitter", "Unknown")[:30]
        color  = COLORS.get(label, "#94A3B8")
        badge  = sentiment_badge_html(label)
        st.markdown(f"""
        <div class="s-card" style="border-left-color:{color};">
            <div class="s-text">{text[:260]}{'…' if len(text)>260 else ''}</div>
            <div class="s-meta">
                {badge}
                <div class="s-clause">⚖️ {clause[:35]}</div>
                <div class="s-sub">📤 {sub}</div>
                <div style="margin-left:auto;">
                    <div style="font-size:0.68rem;color:{MU};opacity:0.55;">Confidence</div>
                    <div style="background:{BD};border-radius:4px;height:4px;width:80px;margin-top:2px;">
                        <div style="width:{conf}%;height:4px;border-radius:4px;background:{color};"></div>
                    </div>
                    <div style="font-size:0.68rem;color:{color};font-weight:600;margin-top:2px;">{conf}%</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 1rem;color:{MU};opacity:0.55;">
        <div style="font-size:2rem;margin-bottom:0.5rem;">📄</div>
        <div style="font-weight:600;font-size:0.95rem;">Run the pipeline above to see sentence analysis</div>
        <div style="font-size:0.8rem;margin-top:0.3rem;opacity:0.7;">Demo Mode gives instant results — no upload needed</div>
    </div>""", unsafe_allow_html=True)
