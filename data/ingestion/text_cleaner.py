"""
IndiaLexABSA — Legal Text Cleaner
===================================
Normalizes raw extracted text from stakeholder PDFs before
sentence segmentation and clause linking.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter


# ─── Regex patterns ───────────────────────────────────────────────────────────
_PAGE_NUM_RE = re.compile(r"(?m)^\s*(?:Page\s+)?\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE)
_HYPHEN_BREAK_RE = re.compile(r"(\w+)-\n(\w+)")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_NBSP_RE = re.compile(r"[\u00a0\u202f\u2009\u200b\ufeff]")
_OCR_FIXES = [
    (re.compile(r"(?<=\d)O(?=\d)"), "0"),
    (re.compile(r"(?i)\bRs\.?\s*"), "₹"),
]
_CITATION_RE = re.compile(r"\(\s*(\d{4})\s*\)\s*(\d+)\s+SCC\s+(\d+)")
_REPEATED_PUNCT_RE = re.compile(r"([.!?]){2,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")


class TextCleaner:
    """Pipeline for normalizing legal PDF text."""

    def __init__(self, remove_urls=True, remove_emails=True, fix_ocr=True, normalize_citations=True):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.fix_ocr = fix_ocr
        self.normalize_citations_flag = normalize_citations

    @staticmethod
    def normalize_unicode(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        return _NBSP_RE.sub(" ", text)

    @staticmethod
    def remove_page_numbers(text: str) -> str:
        return _PAGE_NUM_RE.sub("", text)

    @staticmethod
    def fix_hyphen_breaks(text: str) -> str:
        return _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip()

    @staticmethod
    def fix_ocr_errors(text: str) -> str:
        for pattern, replacement in _OCR_FIXES:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def normalize_citations(text: str) -> str:
        def _fmt(m: re.Match) -> str:
            return f"[{m.group(1)} SCC {m.group(2)}:{m.group(3)}]"
        return _CITATION_RE.sub(_fmt, text)

    @staticmethod
    def clean_repeated_punctuation(text: str) -> str:
        return _REPEATED_PUNCT_RE.sub(r"\1", text)

    @staticmethod
    def remove_headers_footers(text: str) -> str:
        lines = text.split("\n")
        line_counts = Counter(ln.strip() for ln in lines if 3 < len(ln.strip()) < 80)
        repeated = {ln for ln, cnt in line_counts.items() if cnt >= 3}
        return "\n".join(ln for ln in lines if ln.strip() not in repeated)

    def clean(self, text: str) -> str:
        if not text:
            return ""
        text = self.normalize_unicode(text)
        text = self.remove_page_numbers(text)
        text = self.fix_hyphen_breaks(text)
        text = self.remove_headers_footers(text)
        if self.remove_urls:
            text = _URL_RE.sub(" ", text)
        if self.remove_emails:
            text = _EMAIL_RE.sub(" ", text)
        if self.fix_ocr:
            text = self.fix_ocr_errors(text)
        if self.normalize_citations_flag:
            text = self.normalize_citations(text)
        text = self.clean_repeated_punctuation(text)
        return self.normalize_whitespace(text)

    def clean_document(self, doc: dict) -> dict:
        doc = doc.copy()
        doc["full_text"] = self.clean(doc.get("full_text", ""))
        for page in doc.get("pages", []):
            page["text"] = self.clean(page.get("text", ""))
        doc["word_count"] = len(doc["full_text"].split())
        return doc
