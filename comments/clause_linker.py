"""
IndiaLexABSA — Clause Linker
==============================
Links each comment sentence to the most relevant legislation clause
using SBERT semantic similarity + MMR re-ranking.

Pipeline:
  1. Embed sentence (use translated_text if language != 'en')
  2. ChromaDB MMR search → top-5 candidates
  3. Apply confidence threshold filter
  4. Return best match with similarity score

Returns linked sentence dicts ready for labeling.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger
from tqdm import tqdm

from legislation.knowledge_base import KnowledgeBase


DEFAULT_THRESHOLD = 0.35
DEFAULT_TOP_K = 5


class ClauseLinker:
    """Links comment sentences to legislation clauses via semantic similarity."""

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        mmr_lambda: float = 0.7,
    ):
        self.kb = knowledge_base or KnowledgeBase()
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda

    def link_sentence(self, sentence: dict) -> dict:
        """
        Link a single sentence dict to the best matching clause.
        Modifies and returns the sentence dict.
        """
        # Use translated text for non-English sentences
        text = sentence.get("translated_text") or sentence.get("text", "")
        if not text.strip():
            return sentence

        candidates = self.kb.mmr_search(
            query=text,
            top_k=self.top_k,
            candidate_k=self.top_k * 3,
            lambda_=self.mmr_lambda,
        )

        if not candidates:
            return sentence

        best = candidates[0]
        if best["similarity"] >= self.similarity_threshold:
            sentence["clause_id"] = best["clause_id"]
            sentence["similarity_score"] = best["similarity"]
            sentence["clause_title"] = best["title"]
            sentence["top_candidates"] = [
                {"clause_id": c["clause_id"], "title": c["title"], "similarity": c["similarity"]}
                for c in candidates[:3]
            ]
        else:
            sentence["clause_id"] = None
            sentence["similarity_score"] = best["similarity"]

        return sentence

    def link_batch(self, sentences: list[dict], show_progress: bool = True) -> list[dict]:
        """Link a batch of sentences to clauses."""
        linked_count = 0
        iterator = tqdm(sentences, desc="Linking sentences") if show_progress else sentences

        for sent in iterator:
            self.link_sentence(sent)
            if sent.get("clause_id"):
                linked_count += 1

        total = len(sentences)
        logger.info(
            f"Linked {linked_count}/{total} sentences "
            f"({linked_count/total*100:.1f}%) above threshold {self.similarity_threshold}"
        )
        return sentences

    def get_linking_stats(self, sentences: list[dict]) -> dict:
        """Compute statistics on the linking results."""
        linked = [s for s in sentences if s.get("clause_id")]
        unlinked = [s for s in sentences if not s.get("clause_id")]
        scores = [s["similarity_score"] for s in sentences if s.get("similarity_score") is not None]

        clause_counter: dict[str, int] = {}
        for s in linked:
            cid = s["clause_id"]
            clause_counter[cid] = clause_counter.get(cid, 0) + 1

        return {
            "total_sentences": len(sentences),
            "linked": len(linked),
            "unlinked": len(unlinked),
            "link_rate": len(linked) / len(sentences) if sentences else 0,
            "avg_similarity": sum(scores) / len(scores) if scores else 0,
            "min_similarity": min(scores) if scores else 0,
            "max_similarity": max(scores) if scores else 0,
            "top_10_clauses": sorted(clause_counter.items(), key=lambda x: x[1], reverse=True)[:10],
        }
