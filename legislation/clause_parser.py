"""
IndiaLexABSA — Clause Parser
==============================
Parses the Digital Competition Bill (or any Indian legislation) into
a structured hierarchy of sections, sub-sections, clauses, and provisos.

Output per clause:
{
  "clause_id": "S3",
  "section_num": 3,
  "title": "Designation of Systemically Significant Digital Enterprises",
  "text": "...",
  "level": "section",
  "parent_id": null,
  "sub_clauses": ["S3.1", "S3.2"],
  "cross_refs": ["S2", "S7"],
  "proviso": "...",
  "explanation": "..."
}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from loguru import logger


# ─── Regex patterns for Indian legislation structure ──────────────────────────

_SECTION_START = re.compile(
    r"^(\d+)\.\s+(.+?)(?=\n|\.—|\.—|—)",
    re.MULTILINE,
)
_SUBSECTION = re.compile(
    r"^\((\d+)\)\s+(.+?)(?=\n\(|\Z)",
    re.MULTILINE | re.DOTALL,
)
_CLAUSE = re.compile(
    r"^\(([a-z])\)\s+(.+?)(?=\n\([a-z]\)|\n\(\d+\)|\Z)",
    re.MULTILINE | re.DOTALL,
)
_SUB_CLAUSE = re.compile(
    r"^\(([ivxlcIVXLC]+)\)\s+(.+?)(?=\n|\Z)",
    re.MULTILINE,
)
_PROVISO = re.compile(
    r"Provided\s+(?:further\s+)?that[,\s]+(.+?)(?=Provided|Explanation|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_EXPLANATION = re.compile(
    r"Explanation[.\s—:]+(.+?)(?=\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_CROSS_REF = re.compile(
    r"\b[Ss]ection\s+(\d+(?:\(\d+\))?(?:\([a-z]\))?)",
)

# Known section titles for the Digital Competition Bill 2024
DCB_SECTION_TITLES = {
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


@dataclass
class Clause:
    clause_id: str
    section_num: int
    sub_section: Optional[str]
    title: str
    text: str
    level: str          # section / sub_section / clause / sub_clause
    parent_id: Optional[str]
    sub_clauses: list[str] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)
    proviso: str = ""
    explanation: str = ""
    word_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["word_count"] = len(self.text.split())
        return d


class ClauseParser:
    """Parses Indian legislation text into structured clauses."""

    def __init__(self, legislation_text: str = "", section_titles: Optional[dict] = None):
        self.legislation_text = legislation_text
        self.section_titles = section_titles or DCB_SECTION_TITLES
        self.clauses: list[Clause] = []

    def load_text(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.legislation_text = f.read()
        logger.info(f"Loaded legislation text: {len(self.legislation_text)} chars")

    def _extract_cross_refs(self, text: str) -> list[str]:
        return list(set(_CROSS_REF.findall(text)))

    def _extract_proviso(self, text: str) -> tuple[str, str]:
        proviso_match = _PROVISO.search(text)
        explanation_match = _EXPLANATION.search(text)
        proviso = proviso_match.group(1).strip() if proviso_match else ""
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        # Remove proviso/explanation from main text
        clean = _PROVISO.sub("", text)
        clean = _EXPLANATION.sub("", clean).strip()
        return clean, proviso, explanation

    def parse(self) -> list[Clause]:
        """Parse the full legislation text into a list of Clause objects."""
        self.clauses = []

        if not self.legislation_text:
            logger.warning("No legislation text loaded — generating stubs from known titles")
            return self._generate_stubs()

        # Split into sections by "N. Title\n" pattern
        sections = _SECTION_START.split(self.legislation_text)

        i = 1
        while i < len(sections) - 2:
            try:
                sec_num = int(sections[i])
                sec_header = sections[i + 1].strip()
                sec_body = sections[i + 2] if i + 2 < len(sections) else ""

                title = self.section_titles.get(sec_num, sec_header)
                clean_body, proviso, explanation = self._extract_proviso(sec_body)
                cross_refs = self._extract_cross_refs(sec_body)

                section_clause = Clause(
                    clause_id=f"S{sec_num}",
                    section_num=sec_num,
                    sub_section=None,
                    title=title,
                    text=clean_body[:2000],  # cap for embedding
                    level="section",
                    parent_id=None,
                    cross_refs=[f"S{r}" for r in cross_refs if r.isdigit()],
                    proviso=proviso,
                    explanation=explanation,
                )
                self.clauses.append(section_clause)

                # Parse sub-sections
                for ss_match in _SUBSECTION.finditer(sec_body):
                    ss_num = ss_match.group(1)
                    ss_text = ss_match.group(2).strip()
                    ss_id = f"S{sec_num}.{ss_num}"
                    section_clause.sub_clauses.append(ss_id)

                    ss_clause = Clause(
                        clause_id=ss_id,
                        section_num=sec_num,
                        sub_section=ss_num,
                        title=f"{title} ({ss_num})",
                        text=ss_text[:1000],
                        level="sub_section",
                        parent_id=f"S{sec_num}",
                        cross_refs=self._extract_cross_refs(ss_text),
                    )
                    self.clauses.append(ss_clause)

            except (ValueError, IndexError):
                pass
            i += 3

        if not self.clauses:
            return self._generate_stubs()

        logger.info(f"Parsed {len(self.clauses)} clauses from legislation text")
        return self.clauses

    def _generate_stubs(self) -> list[Clause]:
        """Generate clause stubs from known section titles (demo mode)."""
        stubs = []
        for sec_num, title in self.section_titles.items():
            stub = Clause(
                clause_id=f"S{sec_num}",
                section_num=sec_num,
                sub_section=None,
                title=title,
                text=f"Section {sec_num}: {title}. [Full text not loaded — add legislation PDF to data/raw/]",
                level="section",
                parent_id=None,
                cross_refs=[],
                proviso="",
                explanation="",
            )
            stubs.append(stub)
        logger.info(f"Generated {len(stubs)} clause stubs (demo mode)")
        self.clauses = stubs
        return stubs

    def save(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [c.to_dict() for c in self.clauses]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} clauses → {output_path}")

    def load_from_json(self, path: str) -> list[Clause]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.clauses = []
        for d in data:
            c = Clause(**{k: v for k, v in d.items() if k != "word_count"})
            self.clauses.append(c)
        return self.clauses


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="", help="Path to legislation text file")
    parser.add_argument("--output", default="data/processed/clauses.json")
    args = parser.parse_args()

    cp = ClauseParser()
    if args.text:
        cp.load_text(args.text)
    cp.parse()
    cp.save(args.output)
