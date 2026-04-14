"""
IndiaLexABSA — Context Builder
=================================
Assembles the final (sentence, clause, context) triples for labeling.

For each linked sentence, fetches:
  - The full clause text
  - The clause's plain-English enrichment summary
  - Surrounding sentences (window of ±2) for context
  - Cross-referenced related clauses

Output triple format:
{
  "sent_id": "...",
  "doc_id": "...",
  "submitter": "...",
  "category": "...",
  "sentence": "...",
  "sentence_lang": "en",
  "translated": false,
  "clause_id": "S3",
  "clause_title": "...",
  "clause_text": "...",
  "clause_summary": "...",
  "context_window": [...],
  "related_clauses": [...],
  "similarity_score": 0.72,
  "label": null,
  "confidence": null
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger


class ContextBuilder:
    """Assembles labeled training triples from sentences + knowledge base."""

    def __init__(
        self,
        knowledge_base=None,         # KnowledgeBase instance
        cross_referencer=None,       # CrossReferencer instance
        enrichments: Optional[dict] = None,  # {clause_id: enrichment_dict}
        context_window: int = 2,     # sentences before/after
    ):
        self.kb = knowledge_base
        self.xref = cross_referencer
        self.enrichments = enrichments or {}
        self.context_window = context_window

    def _get_clause_info(self, clause_id: str) -> dict:
        """Fetch clause text and enrichment from KB."""
        if self.kb:
            clause = self.kb.get_clause(clause_id)
            if clause:
                enrichment = self.enrichments.get(clause_id, {})
                return {
                    "clause_text": clause.get("document", ""),
                    "clause_title": clause.get("title", clause_id),
                    "clause_summary": enrichment.get("plain_english", ""),
                    "complexity_score": enrichment.get("complexity_score", 5),
                }
        return {
            "clause_text": "",
            "clause_title": clause_id,
            "clause_summary": "",
            "complexity_score": 5,
        }

    def _get_related(self, clause_id: str) -> list[str]:
        if self.xref:
            return self.xref.get_related(clause_id, depth=1)[:3]
        return []

    def build_triples(
        self,
        sentences: list[dict],
        linked_only: bool = True,
    ) -> list[dict]:
        """
        Build context-enriched training triples from a list of sentence dicts.
        sentences must be ordered by position within each document.
        """
        # Group by doc for context window lookup
        from collections import defaultdict
        doc_sents: dict[str, list[dict]] = defaultdict(list)
        for s in sentences:
            doc_sents[s["doc_id"]].append(s)

        triples = []
        for doc_id, doc_sentences in doc_sents.items():
            # Sort by position
            doc_sentences.sort(key=lambda x: x.get("position", 0))

            for idx, sent in enumerate(doc_sentences):
                if linked_only and not sent.get("clause_id"):
                    continue

                clause_id = sent.get("clause_id", "")
                clause_info = self._get_clause_info(clause_id) if clause_id else {}

                # Build context window
                start = max(0, idx - self.context_window)
                end = min(len(doc_sentences), idx + self.context_window + 1)
                context = [
                    {"text": doc_sentences[i]["text"], "position": i, "is_target": i == idx}
                    for i in range(start, end)
                ]

                triple = {
                    "sent_id": sent["sent_id"],
                    "doc_id": doc_id,
                    "submitter": sent.get("submitter", "Unknown"),
                    "category": sent.get("category", "individual"),
                    "sentence": sent.get("translated_text") or sent.get("text", ""),
                    "sentence_original": sent.get("text", ""),
                    "sentence_lang": sent.get("language", "en"),
                    "was_translated": sent.get("translated_text") is not None,
                    "clause_id": clause_id,
                    "clause_title": clause_info.get("clause_title", ""),
                    "clause_text": clause_info.get("clause_text", ""),
                    "clause_summary": clause_info.get("clause_summary", ""),
                    "complexity_score": clause_info.get("complexity_score", 5),
                    "related_clauses": self._get_related(clause_id) if clause_id else [],
                    "context_window": context,
                    "similarity_score": sent.get("similarity_score"),
                    "label": sent.get("label"),
                    "confidence": sent.get("confidence"),
                    "label_source": sent.get("label_source"),
                }
                triples.append(triple)

        logger.info(f"Built {len(triples)} context triples from {len(sentences)} sentences")
        return triples

    def save(self, triples: list[dict], output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for triple in triples:
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(triples)} triples → {output_path}")

    @staticmethod
    def load(path: str) -> list[dict]:
        triples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    triples.append(json.loads(line))
        return triples
