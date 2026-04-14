"""
IndiaLexABSA — PyMuPDF + pdfplumber Hybrid PDF Extractor
=========================================================
Extracts structured text from stakeholder PDFs.
Strategy:
  1. Try PyMuPDF (fitz) first — fast, handles digital PDFs well
  2. Fall back to pdfplumber for complex layouts / tables
  3. Detect and flag scanned pages (low text density)
  4. Return structured JSONL with page metadata

Output schema per document:
{
  "doc_id": "sha256[:12]",
  "filename": "...",
  "submitter": "...",
  "category": "...",
  "pages": [
    {
      "page_num": 1,
      "text": "...",
      "tables": [...],
      "is_scanned": false
    }
  ],
  "full_text": "...",
  "word_count": 1234,
  "extraction_method": "pymupdf"
}

Usage:
    python data/ingestion/pdf_extractor.py \
        --registry data/raw/registry.json \
        --pdf_dir data/raw/ \
        --output data/processed/
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from loguru import logger
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────────
MIN_CHARS_PER_PAGE = 100   # Below this → likely scanned page
SCANNED_CHAR_RATIO = 0.1   # chars/pixel below this → flag scanned


# ─── Core Extractor ──────────────────────────────────────────────────────────

class PDFExtractor:
    """Hybrid PDF text extractor."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─── PyMuPDF extraction ───────────────────────────────────────────────────

    def _extract_pymupdf(self, pdf_path: Path) -> tuple[list[dict], str]:
        """Extract text page-by-page using PyMuPDF."""
        pages = []
        full_text_parts = []

        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                blocks = page.get_text("blocks")
                is_scanned = len(text.strip()) < MIN_CHARS_PER_PAGE

                # Extract tables if any (basic block grouping)
                tables = []
                for block in blocks:
                    if block[6] == 1:  # image block → likely table in scanned doc
                        tables.append({"type": "image_block", "bbox": block[:4]})

                pages.append({
                    "page_num": page_num,
                    "text": text,
                    "tables": tables,
                    "is_scanned": is_scanned,
                    "char_count": len(text),
                })
                full_text_parts.append(text)

            doc.close()
        except Exception as exc:
            logger.error(f"PyMuPDF failed for {pdf_path}: {exc}")
            return [], ""

        return pages, "\n".join(full_text_parts)

    # ─── pdfplumber extraction ────────────────────────────────────────────────

    def _extract_pdfplumber(self, pdf_path: Path) -> tuple[list[dict], str]:
        """Extract text using pdfplumber (better for complex layouts)."""
        pages = []
        full_text_parts = []

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    is_scanned = len(text.strip()) < MIN_CHARS_PER_PAGE

                    # Extract tables
                    raw_tables = page.extract_tables()
                    tables = []
                    for tbl in (raw_tables or []):
                        tables.append({
                            "type": "table",
                            "rows": tbl,
                        })

                    pages.append({
                        "page_num": page_num,
                        "text": text,
                        "tables": tables,
                        "is_scanned": is_scanned,
                        "char_count": len(text),
                    })
                    full_text_parts.append(text)
        except Exception as exc:
            logger.error(f"pdfplumber failed for {pdf_path}: {exc}")
            return [], ""

        return pages, "\n".join(full_text_parts)

    # ─── Quality assessment ───────────────────────────────────────────────────

    @staticmethod
    def _assess_quality(pages: list[dict]) -> dict:
        """Compute quality metrics for extracted text."""
        if not pages:
            return {"avg_chars": 0, "scanned_ratio": 1.0, "total_chars": 0}
        total_chars = sum(p["char_count"] for p in pages)
        scanned_pages = sum(1 for p in pages if p["is_scanned"])
        return {
            "avg_chars": total_chars / len(pages),
            "scanned_ratio": scanned_pages / len(pages),
            "total_chars": total_chars,
            "num_pages": len(pages),
        }

    # ─── Main extraction ─────────────────────────────────────────────────────

    def extract(self, pdf_path: Path, registry_entry: Optional[dict] = None) -> Optional[dict]:
        """Extract text from a PDF, choosing the best method."""
        if not pdf_path.exists():
            logger.warning(f"File not found: {pdf_path}")
            return None

        # Try PyMuPDF first
        pages, full_text = self._extract_pymupdf(pdf_path)
        method = "pymupdf"
        quality = self._assess_quality(pages)

        # Fall back to pdfplumber if quality is poor
        if quality["avg_chars"] < MIN_CHARS_PER_PAGE or quality["scanned_ratio"] > 0.5:
            logger.info(f"Switching to pdfplumber for {pdf_path.name}")
            pages_pb, full_text_pb = self._extract_pdfplumber(pdf_path)
            quality_pb = self._assess_quality(pages_pb)
            if quality_pb["avg_chars"] > quality["avg_chars"]:
                pages, full_text, method = pages_pb, full_text_pb, "pdfplumber"
                quality = quality_pb

        if not full_text.strip():
            logger.warning(f"No text extracted from {pdf_path.name} — may be fully scanned")
            method = "failed_scanned"

        # Build document ID
        doc_id = hashlib.sha256(pdf_path.name.encode()).hexdigest()[:12]

        result = {
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "submitter": (registry_entry or {}).get("submitter", "Unknown"),
            "category": (registry_entry or {}).get("category", "individual"),
            "url": (registry_entry or {}).get("url", ""),
            "pages": pages,
            "full_text": full_text,
            "word_count": len(full_text.split()),
            "char_count": quality["total_chars"],
            "num_pages": quality.get("num_pages", len(pages)),
            "scanned_ratio": quality["scanned_ratio"],
            "extraction_method": method,
        }

        return result

    def extract_and_save(self, pdf_path: Path, registry_entry: Optional[dict] = None) -> Optional[str]:
        """Extract and save as JSONL. Returns output path."""
        result = self.extract(pdf_path, registry_entry)
        if result is None:
            return None

        out_path = self.output_dir / f"{result['doc_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return str(out_path)

    def process_registry(self, registry_path: str, pdf_dir: str) -> list[dict]:
        """Process all PDFs listed in the registry."""
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)

        pdf_dir = Path(pdf_dir)
        results = []

        for entry in tqdm(registry, desc="Extracting PDFs"):
            pdf_path = pdf_dir / entry["filename"]
            out_path = self.extract_and_save(pdf_path, entry)
            if out_path:
                entry["extracted"] = True
                entry["output_path"] = out_path
                results.append(entry)
            else:
                entry["extracted"] = False
                results.append(entry)

        # Update registry
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

        logger.info(f"Processed {len(results)}/{len(registry)} PDFs")
        return results


# ─── Demo sample data generator ──────────────────────────────────────────────

def generate_demo_samples(output_dir: str) -> None:
    """Generate realistic mock extracted documents for demo mode."""
    samples = [
        {
            "doc_id": "demo_001",
            "filename": "google_dcb_submission.pdf",
            "submitter": "Google India Pvt Ltd",
            "category": "startup",
            "url": "",
            "full_text": (
                "Section 3 of the Digital Competition Bill imposes significant compliance burdens "
                "on Systemically Significant Digital Enterprises (SSDEs). We strongly oppose the "
                "mandatory data sharing provisions under Section 12 as they would compromise user "
                "privacy and trade secrets. The definition of SSDE under Section 2(1)(e) is overly "
                "broad and should be narrowed to prevent regulatory overreach. We support the "
                "objectives of promoting fair competition outlined in Section 7. The monetary "
                "thresholds for SSDE designation under Section 3(2) should be revised upward to "
                "align with international standards such as the EU Digital Markets Act."
            ),
            "pages": [{"page_num": 1, "text": "", "tables": [], "is_scanned": False, "char_count": 500}],
            "word_count": 98,
            "num_pages": 8,
            "scanned_ratio": 0.0,
            "extraction_method": "demo",
        },
        {
            "doc_id": "demo_002",
            "filename": "nasscom_dcb_response.pdf",
            "submitter": "NASSCOM",
            "category": "industry_body",
            "url": "",
            "full_text": (
                "NASSCOM welcomes the Digital Competition Bill as a necessary step toward ensuring "
                "fair competitive practices in the digital economy. The provisions under Section 7 "
                "regarding anti-steering restrictions are broadly supported by our members. However, "
                "Section 4 on mandatory interoperability raises serious technical feasibility concerns. "
                "We suggest that the CCI be granted additional investigative powers under Section 16 "
                "to effectively monitor SSDE conduct. Section 12 data sharing obligations require "
                "clearer data localisation safeguards before implementation."
            ),
            "pages": [{"page_num": 1, "text": "", "tables": [], "is_scanned": False, "char_count": 480}],
            "word_count": 85,
            "num_pages": 12,
            "scanned_ratio": 0.0,
            "extraction_method": "demo",
        },
        {
            "doc_id": "demo_003",
            "filename": "cyril_amarchand_mangaldas_submission.pdf",
            "submitter": "Cyril Amarchand Mangaldas",
            "category": "law_firm",
            "url": "",
            "full_text": (
                "The definition of 'digital market' under Section 2(1)(g) is ambiguous and requires "
                "legislative clarification to avoid litigation uncertainty. Section 5 imposes per se "
                "prohibitions that are disproportionate without case-by-case analysis. The mandatory "
                "self-assessment framework under Section 9 is constructive and we support its "
                "implementation. Section 47 sunset clause timeline of 3 years is insufficient for "
                "compliance planning. We strongly recommend that the appeal mechanism under Section 40 "
                "align with established CCI procedures under the Competition Act 2002."
            ),
            "pages": [{"page_num": 1, "text": "", "tables": [], "is_scanned": False, "char_count": 520}],
            "word_count": 90,
            "num_pages": 15,
            "scanned_ratio": 0.0,
            "extraction_method": "demo",
        },
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        out_path = output_dir / f"{sample['doc_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)

    logger.info(f"Generated {len(samples)} demo samples → {output_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PDF text extractor")
    parser.add_argument("--registry", default="data/raw/registry.json")
    parser.add_argument("--pdf_dir", default="data/raw/")
    parser.add_argument("--output", default="data/processed/")
    parser.add_argument("--demo", action="store_true", help="Generate demo sample data")
    args = parser.parse_args()

    if args.demo:
        generate_demo_samples(args.output)
        return

    extractor = PDFExtractor(args.output)
    extractor.process_registry(args.registry, args.pdf_dir)


if __name__ == "__main__":
    main()
