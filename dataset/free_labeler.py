"""
IndiaLexABSA — Free LLM Labeler (Gemini / Groq)
============================================================
Generates silver (automatically labeled) training data using FREE APIs.
No paid API keys required!

Supported backends:
  1. Google Gemini Flash  — FREE, no credit card (aistudio.google.com)
  2. Groq (Llama 3.3 70B) — FREE, no credit card (console.groq.com)

Setup:
  pip install google-generativeai groq

  # Add ONE of these to your .env file:
  GEMINI_API_KEY=your_key_here      # from aistudio.google.com
  GROQ_API_KEY=your_key_here        # from console.groq.com
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger


# ── Prompt templates ──────────────────────────────────────────

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

Respond ONLY in valid JSON (no markdown, no backticks):
{{
  "reasoning": "<one sentence explaining your classification>",
  "label": "<supportive|critical|suggestive|neutral|ambiguous>",
  "confidence": <integer 1-5, where 5 is most confident>
}}"""


VALID_LABELS = {"supportive", "critical", "suggestive", "neutral", "ambiguous"}


# ── Backend: Google Gemini (FREE) ─────────────────────────────

class GeminiBackend:
    """Google Gemini Flash — FREE tier, no credit card."""

    def __init__(self):
        try:
            import google.generativeai as genai
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("No GEMINI_API_KEY")
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.name = "gemini-1.5-flash"
            logger.info("✅ Gemini Flash backend ready (FREE)")
        except Exception as e:
            raise RuntimeError(f"Gemini init failed: {e}")

    def label(self, system: str, prompt: str) -> dict:
        response = self.model.generate_content(
            f"{system}\n\n{prompt}",
            generation_config={"temperature": 0.0, "max_output_tokens": 200},
        )
        text = response.text.strip()
        # Clean markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)


# ── Backend: Groq (FREE) ─────────────────────────────────────

class GroqBackend:
    """Groq Llama 3.3 70B — FREE tier, no credit card."""

    def __init__(self):
        try:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("No GROQ_API_KEY")
            self.client = Groq(api_key=key)
            self.name = "llama-3.3-70b-versatile"
            logger.info("✅ Groq (Llama 3.3 70B) backend ready (FREE)")
        except Exception as e:
            raise RuntimeError(f"Groq init failed: {e}")

    def label(self, system: str, prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)


# ── Backend: OpenAI (Paid fallback) ──────────────────────────

class OpenAIBackend:
    """OpenAI GPT-4o-mini — paid, needs OPENAI_API_KEY."""

    def __init__(self):
        try:
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("No OPENAI_API_KEY")
            self.client = OpenAI(api_key=key)
            self.name = "gpt-4o-mini"
            logger.info("✅ OpenAI (GPT-4o-mini) backend ready (PAID)")
        except Exception as e:
            raise RuntimeError(f"OpenAI init failed: {e}")

    def label(self, system: str, prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=150,
        )
        return json.loads(response.choices[0].message.content)


# ── Auto-detect best free backend ────────────────────────────

def get_backend():
    """Try free backends first, then paid."""
    # Priority: Gemini (free) → Groq (free) → OpenAI (paid)
    for BackendClass, env_var in [
        (GeminiBackend, "GEMINI_API_KEY"),
        (GroqBackend, "GROQ_API_KEY"),
        (OpenAIBackend, "OPENAI_API_KEY"),
    ]:
        if os.getenv(env_var):
            try:
                return BackendClass()
            except Exception as e:
                logger.warning(f"{BackendClass.__name__} failed: {e}")
    return None


# ── Main Labeler Class ───────────────────────────────────────

class FreeLLMLabeler:
    """Silver labeler using the best available FREE LLM."""

    def __init__(self, batch_size: int = 20):
        self.batch_size = batch_size
        self.backend = get_backend()
        self.total_labeled = 0
        self.errors = 0

        if not self.backend:
            logger.error(
                "❌ No LLM backend available!\n"
                "   Set ONE of these in your .env file:\n"
                "   GEMINI_API_KEY=xxx   (free from aistudio.google.com)\n"
                "   GROQ_API_KEY=xxx     (free from console.groq.com)"
            )

    def label_triple(self, triple: dict) -> dict:
        """Label a single (sentence, clause, sentiment) triple."""
        if not self.backend:
            return {**triple, "label": None, "label_source": "unavailable"}

        prompt = LABEL_PROMPT.format(
            clause_title=triple.get("clause_title", "Unknown clause"),
            clause_text=(triple.get("clause_text", "") or "")[:500],
            sentence=triple.get("sentence", ""),
            category=triple.get("category", "individual"),
        )

        for attempt in range(3):
            try:
                result = self.backend.label(LABEL_SYSTEM, prompt)
                label = result.get("label", "neutral").lower().strip()
                if label not in VALID_LABELS:
                    label = "neutral"

                self.total_labeled += 1
                return {
                    **triple,
                    "label": label,
                    "confidence": result.get("confidence", 3),
                    "reasoning": result.get("reasoning", ""),
                    "label_source": self.backend.name,
                }
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)  # exponential backoff

        self.errors += 1
        return {**triple, "label": "neutral", "confidence": 1, "label_source": "fallback"}

    def label_batch(self, triples: list[dict], output_path: str = "") -> list[dict]:
        """Label all triples with progress tracking."""
        from tqdm import tqdm

        labeled = []
        output_file = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output_file = open(output_path, "w", encoding="utf-8")

        try:
            for triple in tqdm(triples, desc=f"Labeling ({self.backend.name if self.backend else 'none'})"):
                if triple.get("label"):  # Already labeled
                    labeled.append(triple)
                    continue

                result = self.label_triple(triple)
                labeled.append(result)

                if output_file and result.get("label"):
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()

                # Rate limiting (be polite to free APIs)
                time.sleep(0.5)  # 2 req/s — safe for all free tiers
        finally:
            if output_file:
                output_file.close()

        labeled_count = sum(1 for t in labeled if t.get("label"))
        logger.info(
            f"✅ Labeled {labeled_count}/{len(triples)} triples "
            f"using {self.backend.name if self.backend else 'none'}. "
            f"Errors: {self.errors}"
        )
        return labeled


# ── CLI entry point ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IndiaLexABSA — Free LLM Labeler")
    parser.add_argument("--input", required=True, help="Input JSONL file with unlabeled triples")
    parser.add_argument("--output", required=True, help="Output JSONL file for labeled triples")
    parser.add_argument("--limit", type=int, default=0, help="Max sentences to label (0=all)")
    args = parser.parse_args()

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Load triples
    triples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))

    if args.limit > 0:
        triples = triples[:args.limit]

    logger.info(f"Loaded {len(triples)} triples from {args.input}")

    # Label
    labeler = FreeLLMLabeler()
    if labeler.backend:
        results = labeler.label_batch(triples, args.output)
        logger.info(f"Done! Output saved to {args.output}")
    else:
        logger.error("No backend available. Set GEMINI_API_KEY or GROQ_API_KEY in .env")
