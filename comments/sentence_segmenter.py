"""
IndiaLexABSA — Sentence Segmenter
====================================
Splits stakeholder comment text into individual sentences using
spaCy with custom legal sentence boundary rules.

Handles:
  - Numbered lists (1. 2. 3.)
  - Lettered sub-items (a) b) c))
  - Legal quotations and parenthetical clauses
  - Multi-line bullet points
  - Footnotes and citations
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterator

from loguru import logger


# ─── Patterns to PREVENT sentence splits ─────────────────────────────────────
# These patterns indicate a period is NOT a sentence boundary
NO_SPLIT = [
    re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g|viz|cf|viz|No|Art|Sec|Cl|Sub)\.$", re.I),
    re.compile(r"\b[A-Z]\.$"),               # Single-letter abbreviations: U.S.A.
    re.compile(r"\b\d+\.\d+"),               # Decimal numbers: 3.5
    re.compile(r"Section\s+\d+\.\d+", re.I),  # Section references: Section 3.2
]

# Patterns that FORCE a sentence split even without punctuation
FORCE_SPLIT = [
    re.compile(r"(?<=\w)\n(?=[A-Z])"),      # Newline + capital letter
    re.compile(r"(?<=\.)\n(?=\d+\.)"),      # After period, before numbered item
]

# Minimum sentence length (chars) to keep
MIN_SENT_LENGTH = 20


class SentenceSegmenter:
    """
    Legal-aware sentence segmenter.
    Primary: spaCy transformer pipeline.
    Fallback: regex-based segmenter.
    """

    def __init__(self, model: str = "en_core_web_sm", use_spacy: bool = True):
        self.use_spacy = use_spacy
        self.nlp = None
        if use_spacy:
            self._load_spacy(model)

    def _load_spacy(self, model: str) -> None:
        try:
            import spacy
            try:
                self.nlp = spacy.load(model, disable=["ner", "lemmatizer"])
            except OSError:
                logger.warning(f"spaCy model '{model}' not found. Trying 'en_core_web_sm'")
                import subprocess, sys
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            self._add_custom_rules()
            logger.info(f"spaCy loaded: {self.nlp.meta.get('name', model)}")
        except Exception as exc:
            logger.warning(f"spaCy unavailable: {exc} — falling back to regex segmenter")
            self.use_spacy = False

    def _add_custom_rules(self) -> None:
        """Add legal-specific sentence boundary overrides."""
        if self.nlp is None:
            return
        # Prevent splits on common legal abbreviations
        legal_abbrevs = [
            "sec", "cl", "sub-sec", "art", "sch", "prov", "expl",
            "vs", "v", "no", "nos", "para", "viz", "i.e", "e.g", "etc",
            "mr", "mrs", "ms", "dr", "prof",
        ]
        ruler = self.nlp.get_pipe("senter") if "senter" in self.nlp.pipe_names else None
        if ruler is None:
            return
        # SpaCy attribute ruler can be used; here we rely on tokenizer exceptions
        for abbrev in legal_abbrevs:
            self.nlp.tokenizer.add_special_case(
                f"{abbrev}.", [{"ORTH": f"{abbrev}."}]
            )

    # ─── spaCy segmentation ───────────────────────────────────────────────────

    def _segment_spacy(self, text: str) -> list[str]:
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= MIN_SENT_LENGTH]

    # ─── Regex fallback ───────────────────────────────────────────────────────

    @staticmethod
    def _segment_regex(text: str) -> list[str]:
        """Simple regex-based sentence splitter for legal text."""
        # Split on ". " or ".\n" followed by capital or number
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(])", text)
        sentences = []
        for part in parts:
            # Further split on newlines followed by numbers (numbered lists)
            sub = re.split(r"\n(?=\d+\.|\([a-z]\)|\([ivx]+\))", part)
            sentences.extend(sub)
        return [s.strip() for s in sentences if len(s.strip()) >= MIN_SENT_LENGTH]

    # ─── Main API ─────────────────────────────────────────────────────────────

    def segment(self, text: str) -> list[str]:
        """Segment text into sentences."""
        if not text or not text.strip():
            return []
        if self.use_spacy and self.nlp:
            # Process in chunks if text is very long (>100k chars)
            if len(text) > 100_000:
                chunks = [text[i:i+50_000] for i in range(0, len(text), 50_000)]
                sentences = []
                for chunk in chunks:
                    sentences.extend(self._segment_spacy(chunk))
                return sentences
            return self._segment_spacy(text)
        return self._segment_regex(text)

    def segment_document(self, doc: dict) -> list[dict]:
        """
        Segment all sentences in a document and return as list of sentence dicts.
        Each sentence has: sent_id, doc_id, text, page_num, position
        """
        doc_id = doc["doc_id"]
        full_text = doc.get("full_text", "")
        sentences_out = []
        position = 0

        sentences = self.segment(full_text)
        for sent in sentences:
            # Generate stable sentence ID
            sent_hash = hashlib.md5(f"{doc_id}:{sent[:50]}:{position}".encode()).hexdigest()[:12]
            sentences_out.append({
                "sent_id": f"{doc_id}_{sent_hash}",
                "doc_id": doc_id,
                "submitter": doc.get("submitter", "Unknown"),
                "category": doc.get("category", "individual"),
                "text": sent,
                "language": "en",  # language detection done later
                "translated_text": None,
                "clause_id": None,
                "similarity_score": None,
                "label": None,
                "confidence": None,
                "label_source": None,
                "page_num": None,   # page attribution added post-processing
                "position": position,
            })
            position += 1

        return sentences_out
