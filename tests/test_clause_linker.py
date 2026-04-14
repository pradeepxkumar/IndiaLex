"""
IndiaLexABSA — Unit Tests: Clause Linker
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from comments.sentence_segmenter import SentenceSegmenter
from legislation.clause_parser import ClauseParser


def test_clause_parser_generates_stubs():
    cp = ClauseParser()
    clauses = cp.parse()
    assert len(clauses) == 47, f"Expected 47 clause stubs, got {len(clauses)}"


def test_clause_ids_format():
    cp = ClauseParser()
    clauses = cp.parse()
    for c in clauses:
        assert c.clause_id.startswith("S"), f"Bad clause_id: {c.clause_id}"
        assert c.section_num > 0


def test_clause_has_required_fields():
    cp = ClauseParser()
    clauses = cp.parse()
    for c in clauses:
        assert c.title, f"Clause {c.clause_id} has no title"
        assert c.text, f"Clause {c.clause_id} has no text"
        assert c.level in ["section", "sub_section", "clause"]


def test_sentence_segmenter_basic():
    seg = SentenceSegmenter(use_spacy=False)
    text = "Section 3 is important. We oppose it. The bill has 47 sections."
    sents = seg.segment(text)
    assert len(sents) >= 2


def test_sentence_segmenter_empty():
    seg = SentenceSegmenter(use_spacy=False)
    assert seg.segment("") == []
    assert seg.segment("   ") == []


def test_segment_document_adds_metadata():
    seg = SentenceSegmenter(use_spacy=False)
    doc = {
        "doc_id": "test001",
        "submitter": "Test Corp",
        "category": "startup",
        "full_text": "We support the Digital Competition Bill. Section 3 needs revision.",
    }
    sents = seg.segment_document(doc)
    assert len(sents) >= 1
    for s in sents:
        assert s["doc_id"] == "test001"
        assert "sent_id" in s
        assert "text" in s
        assert "position" in s


def test_clause_to_dict():
    cp = ClauseParser()
    clauses = cp.parse()
    d = clauses[0].to_dict()
    assert isinstance(d, dict)
    assert "clause_id" in d
    assert "word_count" in d
