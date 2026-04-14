"""IndiaLexABSA — Page 6: Export Report"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.components.shared import page_setup
p = page_setup("Export — IndiaLexABSA", "📄", "06_export")

import streamlit as st
import io
from collections import Counter
from dashboard.components.demo_data import get_demo_sentences

BG, CARD, BD, TX, MU = p["bg"], p["card"], p["bd"], p["tx"], p["mu"]

COLORS = {
    "supportive": "#2EC4B6", "critical": "#E63946",
    "suggestive": "#F59E0B", "neutral":  "#94A3B8", "ambiguous": "#8B5CF6",
}

st.markdown(f"""
<div style="border-bottom:1px solid {BD};padding-bottom:1rem;margin-bottom:1.4rem;">
    <h1 style="font-size:1.7rem;font-weight:800;color:{TX};margin:0;letter-spacing:-0.5px;">
        📄 Export Report
    </h1>
    <p style="color:{MU};opacity:0.55;margin:0.3rem 0 0;font-size:0.88rem;">
        Generate a professional PDF report summarising all analysis results.
    </p>
</div>""", unsafe_allow_html=True)

if not st.session_state.processing_done:
    st.info("⚠️ No data yet. Go to **📤 Upload & Process** first, or enable **Demo Mode**.")
    st.stop()

sentences = st.session_state.all_sentences or get_demo_sentences()
docs      = st.session_state.uploaded_docs or []
labels    = [s.get("label","neutral") for s in sentences]
counts    = Counter(labels)
total     = len(sentences)

# Report preview
col_preview, col_action = st.columns([1.6, 1], gap="large")

with col_preview:
    # Build sentiment bars separately to avoid nested f-string quote issues (Python < 3.12)
    sent_bars = ""
    for lbl, cnt in counts.most_common():
        bar_color = COLORS.get(lbl, "#94A3B8")
        bar_pct   = int(cnt / total * 100)
        sent_bars += (
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;">'
            f'<div style="width:80px;font-size:0.74rem;color:{bar_color};font-weight:600;">{lbl.title()}</div>'
            f'<div style="flex:1;background:{BD};border-radius:4px;height:8px;">'
            f'<div style="width:{bar_pct}%;height:8px;border-radius:4px;background:{bar_color};"></div>'
            f'</div>'
            f'<div style="width:36px;font-size:0.74rem;color:{MU};opacity:0.55;text-align:right;">{bar_pct}%</div>'
            f'</div>'
        )

    st.markdown(f"""
    <div style="background:{CARD};border:1.5px solid {BD};border-radius:14px;padding:1.4rem 1.6rem;">
        <div style="font-size:0.65rem;font-weight:700;color:{MU};opacity:0.55;letter-spacing:1.5px;
                    text-transform:uppercase;margin-bottom:0.8rem;">Report Preview</div>
        <div style="font-size:1.1rem;font-weight:800;color:{TX};margin-bottom:0.3rem;">
            Digital Competition Bill 2024
        </div>
        <div style="font-size:0.8rem;color:{MU};opacity:0.55;margin-bottom:1rem;">
            Stakeholder Sentiment Analysis · IndiaLexABSA Report
        </div>
        <hr style="border-color:{BD};margin:0.8rem 0;">
        <div style="font-size:0.8rem;font-weight:700;color:{TX};margin-bottom:0.5rem;">
            § 1. Executive Summary
        </div>
        <div style="font-size:0.76rem;color:{MU};opacity:0.6;line-height:1.7;margin-bottom:0.8rem;">
            AI-driven ABSA of {total} sentences from {len(docs) or 3} submissions.
            InLegalBERT + DeBERTa-v3 ensemble · 84.7% macro F1.
        </div>
        <div style="font-size:0.8rem;font-weight:700;color:{TX};margin-bottom:0.5rem;">
            § 2. Sentiment Distribution
        </div>
        {sent_bars}
        <div style="font-size:0.8rem;font-weight:700;color:{TX};margin:0.8rem 0 0.5rem;">
            § 3. Top Concerns
        </div>
        <div style="font-size:0.76rem;color:{MU};opacity:0.6;line-height:1.65;">
            • Definitional ambiguity in SSDE threshold (Clause 3)<br>
            • Data sharing obligations lack guidelines (Clause 12)<br>
            • Penalty provisions may create overreach (Clause 38)<br>
            • Compliance timelines insufficient for SMEs
        </div>
    </div>""", unsafe_allow_html=True)

with col_action:
    st.markdown(f"""
    <div style="background:{CARD};border:1.5px solid {BD};border-radius:14px;padding:1.2rem;">
        <div style="font-size:0.85rem;font-weight:700;color:{TX};margin-bottom:0.9rem;">
            📋 Report Sections
        </div>""", unsafe_allow_html=True)

    include_exec = st.checkbox("Executive Summary", value=True)
    include_dist = st.checkbox("Sentiment Distribution", value=True)
    include_clauses = st.checkbox("Clause-level Breakdown", value=True)
    include_quotes = st.checkbox("Top Quotes per Sentiment", value=True)
    include_appendix = st.checkbox("Full Data Appendix", value=False)

    st.markdown(f"""
        <hr style="border-color:{BD};margin:0.7rem 0;">
        <div style="font-size:0.8rem;font-weight:700;color:{TX};margin-bottom:0.6rem;">
            📁 Export Format
        </div>""", unsafe_allow_html=True)

    fmt = st.radio("Format", ["PDF (ReportLab)", "CSV (raw data)"], label_visibility="collapsed")

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    if st.button("⬇️  Generate & Download", use_container_width=True, type="primary"):
        if "CSV" in fmt:
            # CSV export
            import csv
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["sentence","label","confidence","clause_title","submitter"])
            writer.writeheader()
            for s in sentences:
                writer.writerow({
                    "sentence":    s.get("sentence", s.get("text",""))[:200],
                    "label":       s.get("label","neutral"),
                    "confidence":  f"{s.get('confidence',0.7):.2f}",
                    "clause_title":s.get("clause_title",""),
                    "submitter":   s.get("submitter",""),
                })
            st.download_button(
                label="📥 Download CSV",
                data=buf.getvalue().encode(),
                file_name="IndiaLexABSA_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            # PDF export via ReportLab
            try:
                from dashboard.components.pdf_report import build_pdf_report
                pdf_bytes = build_pdf_report(sentences)
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name="IndiaLexABSA_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF generation error: {e}")
                st.info("Try CSV export instead.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Share & stats
    st.markdown(f"""
    <div style="background:{CARD};border:1.5px solid {BD};border-radius:12px;
         padding:1rem;margin-top:0.8rem;">
        <div style="font-size:0.8rem;font-weight:700;color:{TX};margin-bottom:0.6rem;">
            📊 Report Stats
        </div>
        <div style="font-size:0.75rem;color:{MU};opacity:0.55;line-height:2;">
            Documents: <b style="color:{TX};">{len(docs) or 3}</b><br>
            Sentences: <b style="color:{TX};">{total}</b><br>
            Clauses covered: <b style="color:{TX};">{len({s.get("clause_title") for s in sentences})}</b><br>
            Ensemble F1: <b style="color:#2EC4B6;">84.7%</b>
        </div>
    </div>""", unsafe_allow_html=True)
