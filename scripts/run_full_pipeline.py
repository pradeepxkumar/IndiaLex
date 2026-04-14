"""
IndiaLexABSA — Complete Pipeline (Extract → Label → Train)
============================================================
Single script that runs the entire pipeline locally:
  1. Extract text from all PDFs in data/raw/
  2. Segment into sentences
  3. Link sentences to DCB clauses
  4. Auto-label using rule-based classifier (no API needed)
  5. Build train/val/test splits
  6. Fine-tune InLegalBERT + DeBERTa locally

Usage:
    python scripts/run_full_pipeline.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger

# ── Config ────────────────────────────────────────────────────
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_JSONL  = PROCESSED_DIR / "triples.jsonl"
LABELED_JSONL = PROCESSED_DIR / "silver_labeled.jsonl"
TRAIN_JSONL   = PROCESSED_DIR / "train.jsonl"
VAL_JSONL     = PROCESSED_DIR / "val.jsonl"
TEST_JSONL    = PROCESSED_DIR / "test.jsonl"

VALID_LABELS  = {"supportive", "critical", "suggestive", "neutral", "ambiguous"}

# 47 known DCB clause titles for linking
DCB_CLAUSES = {
    1: "Short title, extent and commencement",
    2: "Definitions",
    3: "Designation of Systemically Significant Digital Enterprises",
    4: "Obligations of SSDEs – Interoperability",
    5: "Obligations of SSDEs – Data sharing",
    6: "Obligations of SSDEs – Anti-steering",
    7: "Obligations of SSDEs – Self-preferencing",
    8: "Obligations of SSDEs – Tying and bundling",
    9: "Self-assessment by SSDEs",
    10: "Compliance report",
    11: "Appointment of compliance officer",
    12: "Data access and sharing obligations",
    13: "Interoperability standards",
    14: "Powers of the Commission",
    15: "Investigation",
    16: "Inquiry into anti-competitive practices",
    17: "Power to call for information",
    18: "Search and seizure",
    19: "Interim relief",
    20: "Orders by Commission",
    21: "Commitment",
    22: "Settlement",
    23: "Leniency",
    24: "Appeals",
    25: "Penalties – general",
    26: "Penalties – SSDEs",
    27: "Enhanced penalties",
    28: "Recovery of penalties",
    29: "Civil liability",
    30: "Compensation",
    31: "Private right of action",
    32: "Mergers and acquisitions",
    33: "Notification threshold",
    34: "Review of combinations",
    35: "Market study",
    36: "Consumer protection interface",
    37: "International cooperation",
    38: "Relationship with sectoral regulators",
    39: "Advisory opinions",
    40: "Appellate Tribunal",
    41: "Powers of Appellate Tribunal",
    42: "Enforcement of orders",
    43: "Offences and prosecution",
    44: "Protection of action in good faith",
    45: "Delegated legislation",
    46: "Power to remove difficulties",
    47: "Repeal and savings",
}


# =====================================================================
# STEP 1: Extract text from PDFs
# =====================================================================
def extract_pdfs() -> list[dict]:
    """Extract text from all PDFs in data/raw/."""
    import fitz  # PyMuPDF

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDFs in {RAW_DIR}")

    documents = []
    for pdf_path in pdfs:
        if "dcb_official" in pdf_path.name.lower():
            logger.info(f"  Skipping legislation text: {pdf_path.name}")
            continue

        try:
            doc = fitz.open(str(pdf_path))
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"
            doc.close()

            if len(full_text.strip()) < 50:
                logger.warning(f"  Very little text in {pdf_path.name}, skipping")
                continue

            documents.append({
                "filename": pdf_path.name,
                "full_text": full_text.strip(),
                "word_count": len(full_text.split()),
            })
            logger.info(f"  ✅ {pdf_path.name}: {len(full_text.split())} words")
        except Exception as e:
            logger.error(f"  ❌ Failed to extract {pdf_path.name}: {e}")

    logger.info(f"Extracted {len(documents)} documents")
    return documents


# =====================================================================
# STEP 2: Segment into sentences
# =====================================================================
def segment_sentences(documents: list[dict]) -> list[dict]:
    """Split document text into individual sentences."""
    # Simple but effective sentence splitter for legal text
    sent_pattern = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'      # Standard sentence boundary
        r'(?<=\.)\s*\n+\s*(?=[A-Z])|'    # Period + newline
        r'\n{2,}'                          # Double newline
    )

    all_sentences = []
    sent_id = 0

    for doc in documents:
        text = doc["full_text"]
        # Clean up common PDF artifacts
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix hyphenation

        raw_sents = sent_pattern.split(text)
        for s in raw_sents:
            s = s.strip()
            # Filter: too short, too long, or just numbers/symbols
            if len(s) < 20 or len(s) > 1000:
                continue
            if len(s.split()) < 5:
                continue

            all_sentences.append({
                "sent_id": f"s_{sent_id:05d}",
                "text": s,
                "filename": doc["filename"],
                "word_count": len(s.split()),
            })
            sent_id += 1

    logger.info(f"Segmented {len(all_sentences)} sentences from {len(documents)} documents")
    return all_sentences


# =====================================================================
# STEP 3: Link sentences to DCB clauses (keyword-based)
# =====================================================================
def link_to_clauses(sentences: list[dict]) -> list[dict]:
    """Link each sentence to the most relevant DCB clause using keyword matching."""

    # Keywords for each clause group
    clause_keywords = {
        2: ["definition", "defined", "means", "interpret"],
        3: ["ssde", "significant", "designation", "threshold", "turnover", "market cap"],
        4: ["interoperability", "interoperable", "compatible"],
        5: ["data sharing", "data access", "share data", "data portability"],
        6: ["anti-steering", "steering", "default", "preference"],
        7: ["self-preferencing", "self preferencing", "own product", "ranking"],
        8: ["tying", "bundling", "tied product", "bundle"],
        9: ["self-assessment", "self assessment", "compliance"],
        10: ["compliance report", "reporting"],
        11: ["compliance officer", "officer"],
        12: ["data access", "sharing obligation", "data sharing"],
        13: ["interoperability standard"],
        14: ["commission", "cci", "competition commission", "powers", "authority"],
        15: ["investigation", "investigate", "inquiry"],
        16: ["anti-competitive", "anticompetitive", "unfair practice"],
        17: ["information", "call for information"],
        19: ["interim", "injunction", "relief"],
        20: ["order", "direction"],
        24: ["appeal", "appellate", "tribunal"],
        25: ["penalty", "fine", "penalise"],
        26: ["penalty", "ssde", "fine"],
        32: ["merger", "acquisition", "combination", "takeover"],
        35: ["market study", "market analysis"],
        36: ["consumer protection", "consumer welfare"],
        37: ["international", "cooperation", "dma", "eu"],
        38: ["sectoral regulator", "trai", "rbi", "sebi"],
        40: ["appellate tribunal", "nclat"],
        47: ["repeal", "savings", "sunset"],
    }

    linked = []
    for sent in sentences:
        text_lower = sent["text"].lower()
        best_clause = None
        best_score = 0

        for sec_num, keywords in clause_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_clause = sec_num

        # If no keyword match, try Section number detection
        if best_clause is None or best_score == 0:
            sec_match = re.search(r'section\s+(\d+)', text_lower)
            if sec_match:
                num = int(sec_match.group(1))
                if num in DCB_CLAUSES:
                    best_clause = num
                    best_score = 1

        # Default fallback: assign to the most general clause
        if best_clause is None:
            best_clause = random.choice([3, 5, 7, 12, 14, 25])

        sent["clause_id"] = f"S{best_clause}"
        sent["clause_title"] = DCB_CLAUSES.get(best_clause, f"Section {best_clause}")
        sent["similarity_score"] = min(0.3 + best_score * 0.15, 0.95)
        linked.append(sent)

    logger.info(f"Linked {len(linked)} sentences to clauses")
    return linked


# =====================================================================
# STEP 4: Auto-label using rule-based sentiment classifier (LOCAL)
# =====================================================================
def label_locally(triples: list[dict]) -> list[dict]:
    """Label triples using keyword/pattern-based rules. No API needed."""

    # Comprehensive keyword patterns for Indian legal/regulatory sentiment
    CRITICAL_WORDS = [
        "oppose", "object", "reject", "against", "concern", "problematic",
        "harmful", "detrimental", "adverse", "disproportionate", "burdensome",
        "overreach", "excessive", "arbitrary", "unconstitutional", "violate",
        "infringe", "undermine", "threaten", "danger", "risk", "fear",
        "unclear", "vague", "ambiguous definition", "lacks clarity",
        "strongly disagree", "not acceptable", "cannot support", "flawed",
        "will stifle", "chilling effect", "anti-competitive", "unfair",
        "discriminatory", "inconsistent", "impractical", "unworkable",
        "too broad", "overly broad", "needs to be reconsidered",
    ]

    SUPPORTIVE_WORDS = [
        "support", "welcome", "appreciate", "commend", "endorse", "agree",
        "in favour", "necessary", "important step", "positive", "beneficial",
        "appropriate", "well-drafted", "comprehensive", "robust framework",
        "good initiative", "rightly", "correctly", "laudable", "effective",
        "we agree", "we support", "this is a welcome", "step in the right",
        "promote competition", "protect consumer", "level playing field",
        "align with global", "consistent with", "in line with",
    ]

    SUGGESTIVE_WORDS = [
        "suggest", "recommend", "propose", "amend", "modify", "revise",
        "should be", "could be", "may consider", "urge", "request",
        "we suggest", "we recommend", "should include", "should provide",
        "needs amendment", "should be amended", "consider adding",
        "alternative approach", "better approach", "instead of",
        "we propose", "would benefit from", "should be revised",
        "clarification needed", "should clarify", "needs clarification",
        "safeguards", "additional provision", "exemption for",
    ]

    NEUTRAL_WORDS = [
        "provides that", "states that", "according to", "as per",
        "the bill defines", "section states", "clause provides",
        "for the purposes of", "in this act", "shall mean",
        "notwithstanding", "subject to", "in accordance with",
        "the committee noted", "it was observed", "the provision",
    ]

    labeled = []
    for triple in triples:
        text_lower = triple["text"].lower()

        # Count matches for each category
        critical_score = sum(2 if phrase in text_lower else 0 for phrase in CRITICAL_WORDS)
        supportive_score = sum(2 if phrase in text_lower else 0 for phrase in SUPPORTIVE_WORDS)
        suggestive_score = sum(2 if phrase in text_lower else 0 for phrase in SUGGESTIVE_WORDS)
        neutral_score = sum(2 if phrase in text_lower else 0 for phrase in NEUTRAL_WORDS)

        # Additional heuristics for legal text
        if any(w in text_lower for w in ["however", "but", "although", "while"]):
            critical_score += 1
        if any(w in text_lower for w in ["must", "shall not", "prohibited"]):
            critical_score += 1
        if "?" in triple["text"]:
            suggestive_score += 1

        scores = {
            "critical": critical_score,
            "supportive": supportive_score,
            "suggestive": suggestive_score,
            "neutral": neutral_score,
        }
        max_score = max(scores.values())

        if max_score == 0:
            # No clear signal — use length heuristic
            if len(triple["text"].split()) > 30:
                label = "neutral"
            else:
                label = random.choice(["neutral", "ambiguous"])
            confidence = 2
        elif list(scores.values()).count(max_score) > 1:
            label = "ambiguous"
            confidence = 2
        else:
            label = max(scores, key=scores.get)
            confidence = min(max_score, 5)

        triple["label"] = label
        triple["confidence"] = confidence
        triple["label_source"] = "rule_based"
        labeled.append(triple)

    # Log distribution
    from collections import Counter
    dist = Counter(t["label"] for t in labeled)
    logger.info(f"✅ Labeled {len(labeled)} sentences locally")
    logger.info(f"  Distribution: {dict(dist)}")
    return labeled


# =====================================================================
# STEP 5: Build train/val/test splits
# =====================================================================
def build_splits(labeled: list[dict], train_ratio=0.7, val_ratio=0.15):
    """Split labeled data into train/val/test."""
    random.seed(42)
    random.shuffle(labeled)

    n = len(labeled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = labeled[:train_end]
    val = labeled[train_end:val_end]
    test = labeled[val_end:]

    for split, path in [(train, TRAIN_JSONL), (val, VAL_JSONL), (test, TEST_JSONL)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# =====================================================================
# STEP 6: Train InLegalBERT + DeBERTa locally
# =====================================================================
def train_model(model_name: str, model_id: str, train_data: list, val_data: list):
    """Fine-tune a transformer model on the labeled data."""
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from torch.utils.data import Dataset

    label2id = {l: i for i, l in enumerate(sorted(VALID_LABELS))}
    id2label = {i: l for l, i in label2id.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training {model_name} on {device}")
    logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Dataset class
    class SentimentDataset(Dataset):
        def __init__(self, data):
            self.data = data
            self.encodings = tokenizer(
                [d["text"] for d in data],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(label2id.get(self.data[idx]["label"], 3))
            return item

    train_dataset = SentimentDataset(train_data)
    val_dataset = SentimentDataset(val_data)

    # Training config — optimized for CPU/low-end GPU
    ckpt_dir = ROOT / "models" / "checkpoints" / f"{model_name}_absa"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Adjust batch size for available hardware
    batch_size = 4 if device == "cpu" else 16
    epochs = 3 if device == "cpu" else 5
    grad_accum = 4 if device == "cpu" else 1

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        fp16=(device == "cuda"),
        report_to="none",  # No wandb
        dataloader_pin_memory=False,
    )

    # Metrics
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info(f"🚀 Starting training: {model_name}")
    trainer.train()

    # Save best model
    best_dir = ckpt_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info(f"✅ {model_name} saved → {best_dir}")

    # Evaluate
    results = trainer.evaluate()
    logger.info(f"📊 {model_name} results: {results}")
    return results


# =====================================================================
# MAIN: Run everything
# =====================================================================
def main():
    logger.info("=" * 60)
    logger.info("IndiaLexABSA — Full Pipeline")
    logger.info("=" * 60)

    start_time = time.time()

    # Step 1: Extract PDFs
    logger.info("\n📄 STEP 1: Extracting text from PDFs...")
    documents = extract_pdfs()
    if not documents:
        logger.error("No documents extracted. Check data/raw/ folder.")
        return

    # Step 2: Segment sentences
    logger.info("\n✂️ STEP 2: Segmenting into sentences...")
    sentences = segment_sentences(documents)

    # Step 3: Link to clauses
    logger.info("\n🔗 STEP 3: Linking sentences to DCB clauses...")
    triples = link_to_clauses(sentences)

    # Save triples
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info(f"  Saved triples → {OUTPUT_JSONL}")

    # Step 4: Label locally (no API needed)
    logger.info("\n🏷️ STEP 4: Auto-labeling with rule-based classifier...")
    labeled = label_locally(triples)

    # Save labeled data
    with open(LABELED_JSONL, "w", encoding="utf-8") as f:
        for t in labeled:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info(f"  Saved labeled data → {LABELED_JSONL}")

    logger.info(f"  Saved labeled data → {LABELED_JSONL}")

    # Step 5: Build splits
    logger.info("\n📊 STEP 5: Building train/val/test splits...")
    train, val, test = build_splits(labeled)

    # Step 6: Train models
    logger.info("\n🧠 STEP 6: Training models locally...")

    logger.info("\n── Training InLegalBERT ──")
    try:
        ilb_results = train_model(
            "inlegalbert",
            "law-ai/InLegalBERT",
            train, val,
        )
    except Exception as e:
        logger.error(f"InLegalBERT training failed: {e}")
        ilb_results = {"error": str(e)}

    logger.info("\n── Training DeBERTa-v3 ──")
    try:
        deb_results = train_model(
            "deberta",
            "microsoft/deberta-v3-base",
            train, val,
        )
    except Exception as e:
        logger.error(f"DeBERTa training failed: {e}")
        deb_results = {"error": str(e)}

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info(f"  Time: {elapsed/60:.1f} minutes")
    logger.info(f"  PDFs processed: {len(documents)}")
    logger.info(f"  Sentences extracted: {len(sentences)}")
    logger.info(f"  Labeled: {len(labeled)}")
    logger.info(f"  Train/Val/Test: {len(train)}/{len(val)}/{len(test)}")
    logger.info(f"  InLegalBERT: {ilb_results}")
    logger.info(f"  DeBERTa: {deb_results}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
