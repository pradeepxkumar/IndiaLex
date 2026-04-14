"""
IndiaLexABSA — Unit Tests: PDF Extractor
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from data.ingestion.pdf_extractor import PDFExtractor, generate_demo_samples
from data.ingestion.text_cleaner import TextCleaner


@pytest.fixture(tmp_path_factory=pytest.TempPathFactory.from_config)
def demo_docs(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    generate_demo_samples(str(out))
    return out


def test_generate_demo_samples(tmp_path):
    generate_demo_samples(str(tmp_path))
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 3, f"Expected 3 demo files, got {len(files)}"


def test_demo_doc_schema(tmp_path):
    generate_demo_samples(str(tmp_path))
    for f in tmp_path.glob("*.json"):
        with open(f) as fp:
            doc = json.load(fp)
        assert "doc_id" in doc
        assert "full_text" in doc
        assert "submitter" in doc
        assert "category" in doc
        assert len(doc["full_text"]) > 50


def test_text_cleaner_basic():
    cleaner = TextCleaner()
    raw = "This   is  a  test.\n\n\n\nNew paragraph  here."
    cleaned = cleaner.clean(raw)
    assert "  " not in cleaned  # No double spaces
    assert "\n\n\n" not in cleaned  # No triple newlines


def test_text_cleaner_hyphen_break():
    cleaner = TextCleaner()
    raw = "The contrav-\nention of this provision shall attract penalties."
    cleaned = cleaner.clean(raw)
    assert "contravention" in cleaned


def test_text_cleaner_page_numbers():
    cleaner = TextCleaner()
    raw = "Some content here.\n\n3\n\nMore content."
    cleaned = cleaner.clean(raw)
    assert "\n3\n" not in cleaned


def test_text_cleaner_empty():
    cleaner = TextCleaner()
    assert cleaner.clean("") == ""
    assert cleaner.clean("   ") == ""


def test_pdf_extractor_init(tmp_path):
    extractor = PDFExtractor(str(tmp_path))
    assert extractor.output_dir == tmp_path
