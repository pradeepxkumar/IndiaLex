"""
IndiaLexABSA — Word Cloud Generator Component
"""
from __future__ import annotations
import io
from typing import Optional
from loguru import logger

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","shall","this",
    "that","these","those","it","its","we","our","they","their","which",
    "who","what","when","where","not","no","all","any","section","clause","act","bill",
}


def generate_word_cloud_image(
    text: str,
    colormap: str = "Blues",
    width: int = 400,
    height: int = 250,
    max_words: int = 60,
    background_color: str = "white",
) -> Optional[bytes]:
    if not text.strip():
        return None
    try:
        from wordcloud import WordCloud, STOPWORDS as WC_SW
        wc = WordCloud(
            width=width, height=height,
            background_color=background_color,
            colormap=colormap,
            stopwords=STOPWORDS | WC_SW,
            max_words=max_words,
            prefer_horizontal=0.8,
            collocations=False,
            min_font_size=9, max_font_size=52,
        ).generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        logger.warning(f"Word cloud failed: {exc}")
        return None


def generate_multi_cloud(sentences: list[dict], labels=None, colormaps=None, width=350, height=220):
    if labels is None:
        labels = ["critical", "supportive", "suggestive"]
    if colormaps is None:
        colormaps = {"critical": "Reds", "supportive": "GnBu", "suggestive": "Oranges"}
    result = {}
    for label in labels:
        text = " ".join(s.get("sentence", s.get("text", "")) for s in sentences if s.get("label") == label)
        result[label] = generate_word_cloud_image(text, colormap=colormaps.get(label, "Blues"), width=width, height=height)
    return result
