"""
IndiaLexABSA — GPT-4o Few-Shot Classifier
==========================================
Called only for low-confidence ensemble cases (~8–12% of sentences).
Uses 10 curated examples (2 per class) as few-shot demonstrations.
Returns structured JSON with label + confidence.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

LABELS = ["supportive", "critical", "suggestive", "neutral", "ambiguous"]

FEW_SHOT_EXAMPLES = [
    {"sentence": "We strongly oppose the designation thresholds in Section 3 as they are far too low and will capture companies that pose no systemic risk.", "clause": "Section 3 – Designation of SSDEs", "label": "critical"},
    {"sentence": "The mandatory data sharing requirements would fundamentally compromise trade secrets and must be rejected.", "clause": "Section 12 – Data access", "label": "critical"},
    {"sentence": "We welcome the Commission's proposed powers under Section 14 as a necessary safeguard for competitive markets.", "clause": "Section 14 – Powers of the Commission", "label": "supportive"},
    {"sentence": "The self-assessment framework in Section 9 is a sensible, proportionate approach that balances compliance costs.", "clause": "Section 9 – Self-assessment", "label": "supportive"},
    {"sentence": "We recommend including a safe harbour provision for startups with annual revenue below ₹500 crore.", "clause": "Section 7 – Anti-steering", "label": "suggestive"},
    {"sentence": "The definition of digital market in Section 2 should be revised to exclude B2B platforms from its scope.", "clause": "Section 2 – Definitions", "label": "suggestive"},
    {"sentence": "Section 47 provides for a three-year sunset clause for review of the Act.", "clause": "Section 47 – Repeal and savings", "label": "neutral"},
    {"sentence": "The Act designates SSDEs based on user base, revenue, and market concentration thresholds.", "clause": "Section 3 – Designation", "label": "neutral"},
    {"sentence": "While we appreciate the intent of Section 4, its operational implications remain unclear without further regulatory guidance.", "clause": "Section 4 – Interoperability", "label": "ambiguous"},
    {"sentence": "Our members have mixed views on Section 12 — some see merit in data portability while others cite privacy risks.", "clause": "Section 12 – Data access", "label": "ambiguous"},
]

SYSTEM_PROMPT = """You are a legal sentiment classification expert.
Classify stakeholder comment sentences toward legislation clauses into exactly one of:
supportive | critical | suggestive | neutral | ambiguous

Use the provided examples as reference."""

def _build_prompt(sentence: str, clause_context: str) -> str:
    examples = "\n".join(
        f'Sentence: "{ex["sentence"]}"\nClause: {ex["clause"]}\nLabel: {ex["label"]}'
        for ex in FEW_SHOT_EXAMPLES
    )
    return (
        f"Examples:\n{examples}\n\n"
        f"Now classify:\n"
        f"Sentence: \"{sentence}\"\n"
        f"Clause: {clause_context or 'Unknown'}\n\n"
        "Respond only in JSON: {\"label\": \"...\", \"confidence\": <1-5>, \"reasoning\": \"...\"}"
    )


class GPTFewShotClassifier:
    """GPT-4o few-shot classifier for low-confidence ensemble escalations."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = self._get_client()

    def _get_client(self):
        try:
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                return None
            return OpenAI(api_key=key)
        except ImportError:
            return None

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def predict(self, sentence: str, clause_context: str = "") -> dict:
        if not self._client:
            return {"label": "neutral", "confidence": 0.5, "reasoning": "GPT unavailable", "source": "gpt_stub"}

        prompt = _build_prompt(sentence, clause_context)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=100,
        )
        result = json.loads(response.choices[0].message.content)
        raw_conf = result.get("confidence", 3)
        # Normalize 1-5 scale to 0-1
        conf = (int(raw_conf) - 1) / 4 if isinstance(raw_conf, (int, float)) else 0.5
        return {
            "label": result.get("label", "neutral"),
            "confidence": conf,
            "reasoning": result.get("reasoning", ""),
            "source": "gpt_few_shot",
        }
