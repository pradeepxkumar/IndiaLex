"""
IndiaLexABSA — GPT-4o Clause Enricher
=========================================
Enriches each parsed clause with:
  1. Plain-English summary (for non-legal readers)
  2. Key legal concepts and definitions referenced
  3. Potential ambiguity flags
  4. Stakeholder impact assessment

All results are cached to JSON to avoid re-calling GPT.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


def _get_openai_client():
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        return None


ENRICHMENT_CACHE = "data/processed/clause_enrichments.json"

ENRICH_SYSTEM = """You are a senior legal analyst specializing in Indian competition law.
Your task is to analyze provisions of the Digital Competition Bill, 2024.
Be concise, precise, and accessible to non-legal policymakers."""

ENRICH_PROMPT = """Analyze this provision of the Digital Competition Bill:

Section {section_num}: {title}

Full text:
{text}

Respond in JSON format with these exact fields:
{{
  "plain_english": "<2-3 sentence plain-English summary accessible to a non-lawyer>",
  "key_concepts": ["<legal concept 1>", "<legal concept 2>", ...],
  "ambiguities": ["<ambiguity 1 if any>"],
  "stakeholder_impact": {{
    "tech_companies": "<brief impact>",
    "startups": "<brief impact>",
    "consumers": "<brief impact>",
    "government": "<brief impact>"
  }},
  "complexity_score": <integer 1-10, where 10 is most complex>
}}"""


class ClauseEnricher:
    """Enriches clauses with GPT-4o generated summaries and analysis."""

    def __init__(
        self,
        cache_path: str = ENRICHMENT_CACHE,
        model: str = "gpt-4o-mini",
    ):
        self.cache_path = Path(cache_path)
        self.model = model
        self.client = _get_openai_client()
        self._cache: dict[str, dict] = self._load_cache()

        if not self.client:
            logger.warning("OpenAI client not available — enrichment will use stubs")

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call_gpt(self, clause: dict) -> dict:
        prompt = ENRICH_PROMPT.format(
            section_num=clause.get("section_num", ""),
            title=clause.get("title", ""),
            text=clause.get("text", "")[:2000],
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ENRICH_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=500,
        )
        return json.loads(response.choices[0].message.content)

    def _stub_enrichment(self, clause: dict) -> dict:
        """Generate a placeholder enrichment when GPT is unavailable."""
        title = clause.get("title", "This provision")
        return {
            "plain_english": (
                f"Section {clause.get('section_num', '?')} deals with {title.lower()}. "
                "This provision establishes obligations and procedures related to the designated area. "
                "Stakeholders should review compliance requirements carefully."
            ),
            "key_concepts": ["designation", "compliance", "obligation", "enterprise"],
            "ambiguities": ["Definition scope may require regulatory clarification"],
            "stakeholder_impact": {
                "tech_companies": "May impose compliance obligations",
                "startups": "Threshold-dependent applicability",
                "consumers": "Enhanced protections may apply",
                "government": "New regulatory mandate",
            },
            "complexity_score": 5,
        }

    def enrich_clause(self, clause: dict) -> dict:
        """Enrich a single clause, using cache if available."""
        clause_id = clause["clause_id"]
        if clause_id in self._cache:
            return self._cache[clause_id]

        if self.client:
            try:
                enrichment = self._call_gpt(clause)
            except Exception as exc:
                logger.error(f"GPT enrichment failed for {clause_id}: {exc}")
                enrichment = self._stub_enrichment(clause)
        else:
            enrichment = self._stub_enrichment(clause)

        self._cache[clause_id] = enrichment
        self._save_cache()
        return enrichment

    def enrich_all(self, clauses: list[dict], save_back: bool = True) -> list[dict]:
        """Enrich all clauses and optionally write results back to clause dicts."""
        from tqdm import tqdm
        enriched = []
        for clause in tqdm(clauses, desc="Enriching clauses"):
            enrichment = self.enrich_clause(clause)
            if save_back:
                clause = {**clause, "enrichment": enrichment}
            enriched.append(clause)
        return enriched
