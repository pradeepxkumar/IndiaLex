"""
IndiaLexABSA — GPT Silver Labeler
=====================================
Generates silver (automatically labeled) training data using GPT-4o-mini.

Features:
  - Chain-of-Thought reasoning before final label
  - Structured JSON output (label + confidence + rationale)
  - Batch processing with rate limit handling
  - Exponential backoff retry logic
  - Cost tracking (tokens used)

5 sentiment classes:
  supportive  — stakeholder endorses / agrees with the clause
  critical    — stakeholder opposes / criticizes the clause
  suggestive  — stakeholder proposes amendments / improvements
  neutral     — factual / informational, no clear stance
  ambiguous   — mixed or unclear sentiment
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


LABEL_SYSTEM = """You are an expert NLP annotator specializing in Indian regulatory policy.
Your task is to classify the sentiment of a stakeholder comment sentence toward a specific
clause of the Digital Competition Bill, 2024.

Sentiment classes:
- supportive: The stakeholder endorses, agrees with, or welcomes the clause
- critical: The stakeholder opposes, criticizes, or objects to the clause
- suggestive: The stakeholder suggests amendments, modifications, or improvements
- neutral: The sentence is purely factual or informational with no clear stance
- ambiguous: The sentiment is mixed, unclear, or cannot be determined

Always reason step by step before giving your final answer."""

LABEL_PROMPT = """Analyze this stakeholder comment sentence in the context of the given legislation clause.

CLAUSE: {clause_title}
Clause Text: {clause_text}

SENTENCE TO CLASSIFY: "{sentence}"

Submitter type: {category}

Think step by step:
1. What is the submitter saying about the clause?
2. Is there any positive or negative language?
3. Is there a suggestion for change?

Respond only in JSON:
{{
  "reasoning": "<one sentence explaining your classification>",
  "label": "<supportive|critical|suggestive|neutral|ambiguous>",
  "confidence": <integer 1-5, where 5 is most confident>
}}"""

FEW_SHOT_EXAMPLES = [
    {
        "sentence": "We strongly oppose Section 3's designation thresholds as they are unworkably low.",
        "clause": "Section 3 – Designation of SSDEs",
        "label": "critical", "confidence": 5,
    },
    {
        "sentence": "The CCI's proposed powers under Section 14 are reasonable and necessary.",
        "clause": "Section 14 – Powers of the Commission",
        "label": "supportive", "confidence": 5,
    },
    {
        "sentence": "We recommend including a safe harbour provision in Section 7 for startups.",
        "clause": "Section 7 – Anti-steering obligations",
        "label": "suggestive", "confidence": 5,
    },
    {
        "sentence": "Section 12 defines data access obligations for SSDEs.",
        "clause": "Section 12 – Data access",
        "label": "neutral", "confidence": 4,
    },
    {
        "sentence": "While we welcome the intent of Section 4, the operational implications are unclear.",
        "clause": "Section 4 – Interoperability",
        "label": "ambiguous", "confidence": 3,
    },
]


def _get_client():
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        return OpenAI(api_key=key)
    except ImportError:
        return None


class GPTLabeler:
    """Silver labeler using GPT-4o-mini with CoT prompting."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        batch_size: int = 20,
        cost_limit_usd: float = 10.0,
    ):
        self.model = model
        self.batch_size = batch_size
        self.cost_limit = cost_limit_usd
        self.client = _get_client()
        self.total_tokens = 0
        self.estimated_cost = 0.0

        if not self.client:
            logger.warning("OpenAI client unavailable — GPT labeling disabled")

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost in USD for gpt-4o-mini."""
        # gpt-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens
        return tokens * 0.00000015

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def _label_one(self, triple: dict) -> dict:
        """Label a single triple with GPT."""
        prompt = LABEL_PROMPT.format(
            clause_title=triple.get("clause_title", "Unknown clause"),
            clause_text=(triple.get("clause_text", "") or "")[:500],
            sentence=triple.get("sentence", ""),
            category=triple.get("category", "individual"),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": LABEL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=150,
        )

        usage = response.usage
        self.total_tokens += usage.total_tokens
        self.estimated_cost += self._estimate_cost(usage.total_tokens)

        result = json.loads(response.choices[0].message.content)
        return {
            "label": result.get("label", "neutral"),
            "confidence": result.get("confidence", 3),
            "reasoning": result.get("reasoning", ""),
            "label_source": "gpt-4o-mini",
        }

    def label_triple(self, triple: dict) -> dict:
        """Label a single triple, returns updated triple."""
        if not self.client:
            return {**triple, "label": None, "confidence": None, "label_source": "unavailable"}

        if self.estimated_cost > self.cost_limit:
            logger.warning(f"Cost limit ${self.cost_limit} reached. Stopping.")
            return triple

        try:
            result = self._label_one(triple)
            return {**triple, **result}
        except Exception as exc:
            logger.error(f"Labeling failed for {triple.get('sent_id')}: {exc}")
            return triple

    def label_batch(self, triples: list[dict], output_path: str = "") -> list[dict]:
        """Label all triples, saving progress after each batch."""
        from tqdm import tqdm

        labeled = []
        output_file = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_file = open(output_path, "a", encoding="utf-8")

        try:
            for triple in tqdm(triples, desc="GPT labeling"):
                if triple.get("label"):  # Already labeled (e.g., human)
                    labeled.append(triple)
                    continue

                result = self.label_triple(triple)
                labeled.append(result)

                if output_file and result.get("label"):
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()

                # Rate limiting: ~10 req/s for mini
                time.sleep(0.1)
        finally:
            if output_file:
                output_file.close()

        labeled_count = sum(1 for t in labeled if t.get("label"))
        logger.info(
            f"Labeled {labeled_count}/{len(triples)} triples. "
            f"Tokens: {self.total_tokens:,}. "
            f"Cost: ~${self.estimated_cost:.3f}"
        )
        return labeled

    def generate_synthetic(
        self,
        target_class: str,
        clause_id: str,
        clause_title: str,
        n: int = 50,
    ) -> list[dict]:
        """
        Generate synthetic training examples for underrepresented classes.
        Uses curriculum-hard augmentation strategy.
        """
        if not self.client:
            return []

        synth_prompt = (
            f"Generate {n} diverse stakeholder comment sentences that express "
            f"'{target_class}' sentiment toward this legislation provision: "
            f"'{clause_title}' (Clause {clause_id}). "
            f"Each sentence should be realistic, 15-60 words, from different stakeholder perspectives "
            f"(law firms, tech companies, startups, consumer groups). "
            f"Return as a JSON array of strings."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a legal NLP data augmentation expert."},
                    {"role": "user", "content": synth_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
                max_tokens=2000,
            )
            content = json.loads(response.choices[0].message.content)
            sentences = content.get("sentences", content.get("examples", []))
            if isinstance(sentences, list):
                return [
                    {
                        "sent_id": f"synth_{target_class}_{i}",
                        "doc_id": "synthetic",
                        "sentence": s,
                        "clause_id": clause_id,
                        "clause_title": clause_title,
                        "label": target_class,
                        "confidence": 3,
                        "label_source": "synthetic_gpt",
                        "category": "synthetic",
                        "submitter": "Synthetic",
                    }
                    for i, s in enumerate(sentences[:n])
                ]
        except Exception as exc:
            logger.error(f"Synthetic generation failed: {exc}")
        return []
