"""
IndiaLexABSA — ChromaDB Knowledge Base
==========================================
Stores clause embeddings using Sentence-BERT and provides
semantic search for clause linking.

Two encoder backends (auto-selected):
  1. sentence-transformers (local, ~420MB download) — default
  2. HuggingFace Inference API (free, no download) — set HF_API_TOKEN env var
  3. TF-IDF fallback (no models needed) — automatic when both above fail

Full API docs: https://huggingface.co/inference-api
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from loguru import logger


SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "dcb_clauses"
CHROMA_PATH = "data/chromadb"


def _load_encoder(model_id: str):
    """
    Load the best available encoder:
      1. HF Inference API  (if HF_API_TOKEN is set — no download, free)
      2. Local sentence-transformers (if installed — ~420MB download)
      3. TF-IDF keyword fallback (always works, no models needed)
    """
    hf_token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if hf_token:
        try:
            logger.info("Using HuggingFace Inference API for embeddings (no local download)")
            return _HFInferenceEncoder(model_id=model_id, token=hf_token)
        except Exception as e:
            logger.warning(f"HF API encoder failed: {e}")

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local SBERT model: {model_id}")
        return SentenceTransformer(model_id)
    except Exception as e:
        logger.warning(f"sentence-transformers unavailable: {e}")

    logger.warning("Using TF-IDF keyword fallback encoder (reduced accuracy)")
    return _TFIDFFallbackEncoder()


class _HFInferenceEncoder:
    """HuggingFace Inference API encoder — no local model download needed."""

    def __init__(self, model_id: str, token: str):
        self.model_id = model_id
        self.headers = {"Authorization": f"Bearer {token}"}
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        self._dim = 768

    def encode(self, texts: list[str], normalize_embeddings: bool = True, show_progress_bar: bool = False, **kwargs):
        import numpy as np
        import requests
        if isinstance(texts, str):
            texts = [texts]
        result = []
        # Process in batches of 8 (API limit)
        for i in range(0, len(texts), 8):
            batch = texts[i:i+8]
            try:
                resp = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": batch, "options": {"wait_for_model": True}},
                    timeout=30,
                )
                resp.raise_for_status()
                embs = np.array(resp.json())
                # HF returns token embeddings → mean pool
                if embs.ndim == 3:
                    embs = embs.mean(axis=1)
                result.extend(embs)
            except Exception as exc:
                logger.warning(f"HF API call failed: {exc} — using zero vector")
                result.extend([np.zeros(self._dim)] * len(batch))
        arr = np.array(result)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.where(norms == 0, 1, norms)
        if len(texts) == 1:
            return arr[0]
        return arr


class _TFIDFFallbackEncoder:
    """TF-IDF based fallback encoder when no model is available."""

    def __init__(self, dim: int = 256):
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        self._dim = dim
        self._fitted = False
        self._vec = TfidfVectorizer(max_features=dim, ngram_range=(1, 2))
        self._corpus: list[str] = []

    def _ensure_fit(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        all_texts = self._corpus + texts
        self._vec = TfidfVectorizer(max_features=self._dim, ngram_range=(1, 2))
        self._vec.fit(all_texts)
        self._corpus = list(set(all_texts))
        self._fitted = True

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, **kwargs):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        self._ensure_fit(texts)
        sparse = self._vec.transform(texts)
        arr = sparse.toarray().astype(np.float32)
        # Pad or trim to dim
        if arr.shape[1] < self._dim:
            arr = np.pad(arr, ((0,0),(0, self._dim - arr.shape[1])))
        arr = arr[:, :self._dim]
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.where(norms == 0, 1, norms)
        return arr[0] if len(texts) == 1 else arr


class KnowledgeBase:
    """ChromaDB-backed clause knowledge base with SBERT embeddings."""

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        sbert_model: str = SBERT_MODEL,
    ):
        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Auto-selects: HF Inference API → local SBERT → TF-IDF fallback
        self.encoder = _load_encoder(sbert_model)

        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Knowledge base ready: {self.collection.count()} clauses indexed")

    # ─── Indexing ─────────────────────────────────────────────────────────────

    def add_clauses(self, clauses: list[dict], batch_size: int = 32) -> None:
        """Add or update clauses in the collection."""
        if not clauses:
            return

        # Build text for embedding: title + text provides better recall
        texts = [
            f"{c.get('title', '')}. {c.get('text', '')}"[:1024]
            for c in clauses
        ]
        ids = [c["clause_id"] for c in clauses]
        metadatas = [
            {
                "clause_id": c["clause_id"],
                "section_num": str(c.get("section_num", "")),
                "title": c.get("title", "")[:200],
                "level": c.get("level", "section"),
                "parent_id": c.get("parent_id") or "",
                "word_count": str(c.get("word_count", len(c.get("text", "").split()))),
            }
            for c in clauses
        ]

        # Embed in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = self.encoder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.extend(embs.tolist())

        # Upsert to ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=all_embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        logger.info(f"Indexed {len(clauses)} clauses. Total: {self.collection.count()}")

    def load_from_json(self, clauses_path: str) -> None:
        """Load clauses from JSON file and index them."""
        with open(clauses_path, encoding="utf-8") as f:
            clauses = json.load(f)
        self.add_clauses(clauses)

    # ─── Search ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic search returning top_k clauses with scores."""
        if self.collection.count() == 0:
            logger.warning("Knowledge base is empty — run add_clauses() first")
            return []

        query_emb = self.encoder.encode(query, normalize_embeddings=True).tolist()

        kwargs = {
            "query_embeddings": [query_emb],
            "n_results": min(top_k, self.collection.count()),
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        for i, (dist, meta, doc) in enumerate(zip(
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0],
        )):
            # ChromaDB cosine distance → similarity
            similarity = 1.0 - dist
            output.append({
                "rank": i + 1,
                "clause_id": meta["clause_id"],
                "title": meta["title"],
                "level": meta["level"],
                "document": doc,
                "similarity": round(similarity, 4),
            })
        return output

    def mmr_search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 15,
        lambda_: float = 0.7,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance search.
        Balances relevance (lambda_=1.0) vs diversity (lambda_=0.0).
        """
        import numpy as np

        if self.collection.count() == 0:
            return []

        query_emb = self.encoder.encode(query, normalize_embeddings=True)
        candidates = self.search(query, top_k=candidate_k)

        if not candidates:
            return []

        # Re-embed candidates for MMR calculation
        cand_texts = [c["document"] for c in candidates]
        cand_embs = self.encoder.encode(cand_texts, normalize_embeddings=True)

        selected_indices = []
        selected_embs = []

        for _ in range(min(top_k, len(candidates))):
            scores = []
            for j, (cand, emb) in enumerate(zip(candidates, cand_embs)):
                if j in selected_indices:
                    scores.append(-float("inf"))
                    continue
                relevance = float(np.dot(query_emb, emb))
                if selected_embs:
                    redundancy = max(float(np.dot(sel_emb, emb)) for sel_emb in selected_embs)
                else:
                    redundancy = 0.0
                mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
                scores.append(mmr_score)

            best = int(np.argmax(scores))
            selected_indices.append(best)
            selected_embs.append(cand_embs[best])

        return [candidates[i] for i in selected_indices]

    def get_clause(self, clause_id: str) -> Optional[dict]:
        """Retrieve a single clause by ID."""
        try:
            result = self.collection.get(ids=[clause_id], include=["metadatas", "documents"])
            if result["ids"]:
                return {"clause_id": clause_id, **result["metadatas"][0], "document": result["documents"][0]}
        except Exception:
            pass
        return None

    def clear(self) -> None:
        """Clear all indexed clauses."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Knowledge base cleared")
