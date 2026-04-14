"""
IndiaLexABSA — Human Annotation Tool (Streamlit)
==================================================
A lightweight Streamlit app for human annotation of sentence-clause-sentiment triples.
Annotators see: the sentence, the linked clause, a GPT-suggested label,
and choose the final label. Results are saved to SQLite.

Run with:
    streamlit run dataset/human_annotation_tool.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="IndiaLexABSA Annotator", page_icon="🏷️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 900px; }
.card { background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; }
.sentence-box { background: #F7FAFC; border-radius: 8px; padding: 1rem; font-size: 1rem; line-height: 1.7; color: #2D3748; border-left: 4px solid #2EC4B6; }
.clause-box { background: #EBF8FF; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.88rem; color: #2B6CB0; }
.gpt-suggestion { background: #FFF5F5; border-radius: 8px; padding: 0.5rem 1rem; font-size: 0.85rem; color: #C53030; display: inline-block; }
</style>
""", unsafe_allow_html=True)

LABELS = ["supportive", "critical", "suggestive", "neutral", "ambiguous"]
LABEL_EMOJIS = {"supportive": "✅", "critical": "❌", "suggestive": "💡", "neutral": "➖", "ambiguous": "❓"}
DB_PATH = "data/processed/human_annotations.db"
TRIPLES_PATH = "data/processed/triples.jsonl"

# ── DB setup ─────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            sent_id TEXT PRIMARY KEY,
            sentence TEXT,
            clause_id TEXT,
            clause_title TEXT,
            gpt_label TEXT,
            human_label TEXT,
            annotator TEXT,
            timestamp TEXT,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_annotation(record: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO annotations
        (sent_id, sentence, clause_id, clause_title, gpt_label, human_label, annotator, timestamp, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["sent_id"], record["sentence"], record.get("clause_id",""),
        record.get("clause_title",""), record.get("gpt_label",""),
        record["human_label"], record.get("annotator",""),
        datetime.utcnow().isoformat(), record.get("notes",""),
    ))
    conn.commit()
    conn.close()

def get_annotated_ids() -> set:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT sent_id FROM annotations").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        return set()

def get_stats() -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        by_label = conn.execute(
            "SELECT human_label, COUNT(*) FROM annotations GROUP BY human_label"
        ).fetchall()
        conn.close()
        return {"total": total, "by_label": dict(by_label)}
    except Exception:
        return {"total": 0, "by_label": {}}

init_db()
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# ── Load triples ─────────────────────────────────────────────
@st.cache_data
def load_triples():
    if not Path(TRIPLES_PATH).exists():
        # Use demo data
        from dashboard.components.demo_data import get_demo_sentences
        triples = []
        for s in get_demo_sentences(50):
            triples.append({
                "sent_id": s["sent_id"],
                "sentence": s.get("sentence", s.get("text", "")),
                "clause_id": s.get("clause_id",""),
                "clause_title": s.get("clause_title",""),
                "clause_text": "",
                "gpt_label": s.get("label","neutral"),
            })
        return triples
    triples = []
    with open(TRIPLES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                triples.append({
                    "sent_id": t.get("sent_id",""),
                    "sentence": t.get("sentence", t.get("text","")),
                    "clause_id": t.get("clause_id",""),
                    "clause_title": t.get("clause_title",""),
                    "clause_text": t.get("clause_text","")[:200],
                    "gpt_label": t.get("label",""),
                })
    return triples

triples = load_triples()
annotated_ids = get_annotated_ids()
remaining = [t for t in triples if t["sent_id"] not in annotated_ids]

# ── Sidebar stats ─────────────────────────────────────────────
st.sidebar.title("🏷️ Annotation Tool")
stats = get_stats()
progress_pct = stats["total"] / max(len(triples), 1) * 100
st.sidebar.progress(int(progress_pct))
st.sidebar.markdown(f"**{stats['total']}** / {len(triples)} annotated ({progress_pct:.0f}%)")
annotator_name = st.sidebar.text_input("Your Name", value="Annotator")

st.sidebar.markdown("---")
st.sidebar.markdown("**Label Distribution**")
for label in LABELS:
    cnt = stats["by_label"].get(label, 0)
    st.sidebar.markdown(f"{LABEL_EMOJIS.get(label,'●')} **{label.capitalize()}**: {cnt}")

st.sidebar.markdown("---")
if st.sidebar.button("Export Annotations"):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM annotations").fetchall()
    conn.close()
    export_data = [
        {"sent_id": r[0], "sentence": r[1], "clause_id": r[2], "clause_title": r[3],
         "gpt_label": r[4], "label": r[5], "annotator": r[6], "label_source": "human"}
        for r in rows
    ]
    jsonl = "\n".join(json.dumps(d) for d in export_data)
    st.sidebar.download_button("⬇️ Download JSONL", data=jsonl, file_name="human_annotations.jsonl")

# ── Main annotation interface ─────────────────────────────────
st.markdown("# 🏷️ IndiaLexABSA Annotation")
st.markdown("Classify each sentence's sentiment toward the linked legislation clause.")

if not remaining:
    st.success(f"🎉 All {len(triples)} triples annotated! Use the export button in the sidebar.")
    st.stop()

# Initialize index in session state
if "annot_idx" not in st.session_state:
    st.session_state.annot_idx = 0
if st.session_state.annot_idx >= len(remaining):
    st.session_state.annot_idx = 0

triple = remaining[st.session_state.annot_idx]

# Navigation
nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
with nav_col1:
    if st.button("← Prev") and st.session_state.annot_idx > 0:
        st.session_state.annot_idx -= 1
        st.rerun()
with nav_col2:
    st.markdown(f"<div style='text-align:center;color:#718096;font-size:0.85rem;'>"
                f"Triple {st.session_state.annot_idx + 1} of {len(remaining)} remaining</div>",
                unsafe_allow_html=True)
with nav_col3:
    if st.button("Next →") and st.session_state.annot_idx < len(remaining) - 1:
        st.session_state.annot_idx += 1
        st.rerun()

st.markdown("---")

# Sentence display
st.markdown(f"""
<div class="card">
    <div style="font-size:0.72rem;font-weight:600;color:#718096;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.5rem;">
        Sentence to Classify
    </div>
    <div class="sentence-box">{triple['sentence']}</div>
    <div style="margin-top:0.75rem;">
        <b>Linked Clause:</b>
        <div class="clause-box" style="margin-top:0.3rem;">
            <b>{triple.get('clause_id','')}</b>: {triple.get('clause_title','')}
            {f"<br><span style='color:#718096;font-size:0.8rem;'>{triple.get('clause_text','')}</span>" if triple.get('clause_text') else ''}
        </div>
    </div>
    {f"<div style='margin-top:0.75rem;'><b>GPT Suggestion:</b> <span class='gpt-suggestion'>{LABEL_EMOJIS.get(triple.get('gpt_label',''),'?')} {triple.get('gpt_label','').capitalize()}</span></div>" if triple.get('gpt_label') else ''}
</div>
""", unsafe_allow_html=True)

# Label selection
st.markdown("#### Your Label")
label_cols = st.columns(5)
selected_label = None
for i, (col, label) in enumerate(zip(label_cols, LABELS)):
    color_map = {"supportive": "#2EC4B6", "critical": "#E63946", "suggestive": "#F4A261",
                 "neutral": "#8D99AE", "ambiguous": "#9B5DE5"}
    if col.button(
        f"{LABEL_EMOJIS[label]} {label.capitalize()}",
        use_container_width=True,
        key=f"label_btn_{label}",
    ):
        selected_label = label

notes = st.text_input("Notes (optional)", placeholder="Any ambiguity or reasoning...")

if selected_label:
    save_annotation({
        "sent_id": triple["sent_id"],
        "sentence": triple["sentence"],
        "clause_id": triple.get("clause_id",""),
        "clause_title": triple.get("clause_title",""),
        "gpt_label": triple.get("gpt_label",""),
        "human_label": selected_label,
        "annotator": annotator_name,
        "notes": notes,
    })
    st.session_state.annot_idx = min(st.session_state.annot_idx + 1, len(remaining) - 1)
    st.rerun()
