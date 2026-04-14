"""
IndiaLexABSA — Language Handler
==================================
Language detection and Hindi→English translation pipeline.

Architecture:
  1. langdetect → identify language of each sentence
  2. IndicTrans2 (if available) → translate Hindi sentences
  3. Cache all translations to avoid re-running

Fallback: googletrans library if IndicTrans2 is not installed
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from loguru import logger


TRANSLATION_CACHE_PATH = "data/processed/translation_cache.json"
SUPPORTED_LANGS = {"en", "hi", "mr", "gu", "ta", "te", "bn", "kn"}


class LanguageHandler:
    """Detects language and translates non-English sentences."""

    def __init__(self, cache_path: str = TRANSLATION_CACHE_PATH, use_gpu: bool = False):
        self.cache_path = Path(cache_path)
        self.use_gpu = use_gpu
        self._translation_cache: dict[str, str] = self._load_cache()
        self._detector = self._load_detector()
        self._translator = self._load_translator()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._translation_cache, f, ensure_ascii=False, indent=2)

    def _load_detector(self):
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 42
            return detect
        except ImportError:
            logger.warning("langdetect not installed — defaulting all text to English")
            return None

    def _load_translator(self):
        """Load IndicTrans2 or fall back to googletrans."""
        # Try IndicTrans2 first
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
            logger.info(f"Loading IndicTrans2: {model_name}")
            import torch
            device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
            logger.info(f"IndicTrans2 loaded on {device}")
            return {"type": "indictrans2", "tokenizer": tokenizer, "model": model, "device": device}
        except Exception as e:
            logger.warning(f"IndicTrans2 unavailable ({e}), trying googletrans")

        # Fallback: googletrans
        try:
            from googletrans import Translator
            translator = Translator()
            logger.info("Using googletrans as translation fallback")
            return {"type": "googletrans", "translator": translator}
        except ImportError:
            logger.warning("No translation backend available — Hindi sentences will not be translated")
            return None

    def detect_language(self, text: str) -> str:
        """Detect the language of a text string. Returns ISO 639-1 code."""
        if not text or not text.strip():
            return "en"
        if self._detector is None:
            return "en"
        try:
            lang = self._detector(text)
            return lang if lang in SUPPORTED_LANGS else "en"
        except Exception:
            return "en"

    def translate_to_english(self, text: str, source_lang: str = "hi") -> str:
        """Translate text to English, using cache when possible."""
        cache_key = f"{source_lang}:{text[:100]}"
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]

        if self._translator is None:
            return text  # No translator available

        try:
            translation = self._do_translate(text, source_lang)
            self._translation_cache[cache_key] = translation
            self._save_cache()
            return translation
        except Exception as exc:
            logger.error(f"Translation failed: {exc}")
            return text

    def _do_translate(self, text: str, source_lang: str) -> str:
        """Run actual translation based on available backend."""
        if self._translator is None:
            return text

        t_type = self._translator["type"]

        if t_type == "indictrans2":
            tokenizer = self._translator["tokenizer"]
            model = self._translator["model"]
            device = self._translator["device"]
            import torch

            # Format input for IndicTrans2
            src_lang_code = f"{source_lang}_Deva" if source_lang == "hi" else f"{source_lang}_Latn"
            tgt_lang_code = "eng_Latn"

            inputs = tokenizer(
                text,
                src_lang=src_lang_code,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)

            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    tgt_lang=tgt_lang_code,
                    max_new_tokens=256,
                    num_beams=4,
                )
            return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

        elif t_type == "googletrans":
            result = self._translator["translator"].translate(text, src=source_lang, dest="en")
            return result.text

        return text

    def process_sentences(self, sentences: list[dict]) -> list[dict]:
        """Detect language and translate non-English sentences in-place."""
        hi_count = 0
        for sent in sentences:
            text = sent.get("text", "")
            lang = self.detect_language(text)
            sent["language"] = lang

            if lang != "en" and lang in SUPPORTED_LANGS:
                translated = self.translate_to_english(text, lang)
                sent["translated_text"] = translated
                hi_count += 1

        if hi_count:
            logger.info(f"Translated {hi_count}/{len(sentences)} non-English sentences")
        return sentences
