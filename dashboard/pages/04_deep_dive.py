"""IndiaLexABSA — Page 4: Deep Dive"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.components.shared import page_setup
p = page_setup("Deep Dive — IndiaLexABSA", "🔍", "04_deep_dive")

import streamlit as st
import plotly.graph_objects as go
from collections import Counter, defaultdict
from dashboard.components.demo_data import get_demo_sentences
from dashboard.components.sentiment_badge import sentiment_badge_html

BG, CARD, BD, TX, MU = p["bg"], p["card"], p["bd"], p["tx"], p["mu"]

COLORS = {
    "supportive": "#2EC4B6", "critical": "#E63946",
    "suggestive": "#F59E0B", "neutral":  "#94A3B8", "ambiguous": "#8B5CF6",
}

CLAUSE_SUMMARIES = {
    "Clause 3 – Market Definition":      "Defines 'significant digital enterprise' using quantitative thresholds for revenue, user base, and market capitalisation.",
    "Clause 12 – Data Sharing":          "Mandates data sharing obligations for significant digital enterprises with third-party service providers.",
    "Clause 22 – Consumer Protection":   "Prohibits anti-steering provisions and mandates transparent pricing for end consumers.",
    "Clause 15 – Interoperability":      "Requires messaging and payment platforms above threshold to provide interoperability APIs.",
    "Clause 7 – Self-preferencing":      "Prohibits preferential treatment of own products/services in search results, app stores, or marketplaces.",
    "Clause 31 – Dispute Resolution":    "Establishes a dedicated fast-track adjudicatory mechanism for digital competition disputes.",
    "Clause 18 – Platform Neutrality":   "Requires gatekeepers to treat all third-party operators on equal commercial terms.",
    "Clause 9 – Bundling":               "Restricts tying and bundling practices that foreclose competing complementary services.",
}

st.markdown(f"""
<div style="border-bottom:1px solid {BD};padding-bottom:1rem;margin-bottom:1.4rem;">
    <h1 style="font-size:1.7rem;font-weight:800;color:{TX};margin:0;letter-spacing:-0.5px;">
        🔍 Deep Dive
    </h1>
    <p style="color:{MU};opacity:0.55;margin:0.3rem 0 0;font-size:0.88rem;">
        Per-clause analysis: full text, stakeholder reactions, and sentence-level sentiment breakdown.
    </p>
</div>""", unsafe_allow_html=True)

if not st.session_state.processing_done:
    st.info("⚠️ No data yet. Go to **📤 Upload & Process** first, or enable **Demo Mode**.")
    st.stop()

sentences = st.session_state.all_sentences or get_demo_sentences()
clause_options = sorted({s.get("clause_title", "Unknown") for s in sentences})

selected_clause = st.selectbox("Select a clause to analyse:", clause_options)
clause_sents = [s for s in sentences if s.get("clause_title") == selected_clause]

if not clause_sents:
    st.warning("No sentences found for this clause.")
    st.stop()

left_col, center_col, right_col = st.columns([1, 1.4, 1], gap="medium")

# Left: clause info
with left_col:
    summary = CLAUSE_SUMMARIES.get(selected_clause, "This clause addresses key provisions of the Digital Competition Bill 2024.")
    st.markdown(f"""
    <div style="background:{CARD};border:1.5px solid {BD};border-radius:12px;padding:1rem;">
        <div style="font-weight:700;font-size:0.88rem;color:{TX};margin-bottom:0.5rem;">
            ⚖️ {selected_clause}
        </div>
        <div style="font-size:0.8rem;color:{MU};opacity:0.6;line-height:1.65;">{summary}</div>
        <hr style="border-color:{BD};margin:0.75rem 0;">
        <div style="font-size:0.75rem;color:{MU};opacity:0.6;">
            <b style="color:{TX};">Total sentences:</b> {len(clause_sents)}<br>
            <b style="color:{TX};">Languages:</b> English, Hindi<br>
            <b style="color:{TX};">Submitters:</b> {len({s.get('submitter') for s in clause_sents})}
        </div>
    </div>""", unsafe_allow_html=True)

# Center: sentence cards
with center_col:
    lbl_counts = Counter(s.get("label","neutral") for s in clause_sents)
    dom = lbl_counts.most_common(1)[0][0] if lbl_counts else "neutral"
    dom_pct = int(lbl_counts[dom] / len(clause_sents) * 100)

    # Mini donut
    fig = go.Figure(go.Pie(
        labels=[l.title() for l in lbl_counts.keys()],
        values=list(lbl_counts.values()),
        hole=0.65,
        marker_colors=[COLORS.get(l,"#94A3B8") for l in lbl_counts.keys()],
        textinfo="percent",
        textfont_size=11,
    ))
    fig.update_layout(
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=5,b=5,l=5,r=5), height=180,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text=f"<b>{dom_pct}%</b>", x=0.5, y=0.5,
                          font_size=14, showarrow=False, font_color=COLORS.get(dom,"#2EC4B6"))],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div style='font-weight:700;font-size:0.85rem;color:{TX};margin-bottom:0.4rem;'>Sentences ({len(clause_sents)})</div>", unsafe_allow_html=True)
    for s in clause_sents[:5]:
        label  = s.get("label","neutral")
        color  = COLORS.get(label,"#94A3B8")
        text   = s.get("sentence", s.get("text",""))
        conf   = int((s.get("confidence") or 0.7)*100)
        badge  = sentiment_badge_html(label)
        st.markdown(f"""
        <div class="s-card" style="border-left-color:{color};">
            <div class="s-text">{text[:200]}{'…' if len(text)>200 else ''}</div>
            <div class="s-meta">
                {badge}
                <span style="margin-left:auto;font-size:0.68rem;color:{MU};opacity:0.55;">{conf}%</span>
            </div>
        </div>""", unsafe_allow_html=True)

# Right: stakeholder table
with right_col:
    st.markdown(f"<div style='font-weight:700;font-size:0.85rem;color:{TX};margin-bottom:0.5rem;'>Stakeholder Reactions</div>", unsafe_allow_html=True)
    submitters = defaultdict(lambda: {"labels":[], "quotes":[]})
    for s in clause_sents:
        sub = s.get("submitter","Unknown")
        submitters[sub]["labels"].append(s.get("label","neutral"))
        submitters[sub]["quotes"].append(s.get("sentence", s.get("text","")))

    for sub, data in list(submitters.items())[:6]:
        dom_lbl = Counter(data["labels"]).most_common(1)[0][0]
        color   = COLORS.get(dom_lbl,"#94A3B8")
        quote   = data["quotes"][0][:80]+"…" if data["quotes"] else ""
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BD};border-left:3px solid {color};
             border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.4rem;">
            <div style="font-size:0.75rem;font-weight:700;color:{TX};">{sub[:20]}</div>
            <div style="font-size:0.68rem;color:{color};font-weight:600;margin:2px 0;">{dom_lbl.title()}</div>
            <div style="font-size:0.68rem;color:{MU};opacity:0.55;line-height:1.45;">{quote}</div>
        </div>""", unsafe_allow_html=True)
