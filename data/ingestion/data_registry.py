"""
IndiaLexABSA — Data Registry (SQLite-backed)
=============================================
Tracks every PDF through its lifecycle:
  download → extraction → cleaning → linking → labeling
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


DB_PATH = "data/registry.db"


class DataRegistry:
    """SQLite-backed registry for tracking document processing state."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id       TEXT PRIMARY KEY,
                    filename     TEXT NOT NULL,
                    url          TEXT,
                    submitter    TEXT,
                    category     TEXT,
                    sha256       TEXT,
                    size_bytes   INTEGER,
                    num_pages    INTEGER,
                    word_count   INTEGER,
                    language     TEXT DEFAULT 'en',
                    downloaded   INTEGER DEFAULT 0,
                    extracted    INTEGER DEFAULT 0,
                    cleaned      INTEGER DEFAULT 0,
                    linked       INTEGER DEFAULT 0,
                    labeled      INTEGER DEFAULT 0,
                    sentence_count INTEGER DEFAULT 0,
                    created_at   TEXT,
                    updated_at   TEXT
                );

                CREATE TABLE IF NOT EXISTS sentences (
                    sent_id      TEXT PRIMARY KEY,
                    doc_id       TEXT,
                    clause_id    TEXT,
                    text         TEXT NOT NULL,
                    language     TEXT DEFAULT 'en',
                    translated_text TEXT,
                    similarity_score REAL,
                    label        TEXT,
                    confidence   REAL,
                    label_source TEXT,  -- gpt / human / ensemble
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                );

                CREATE INDEX IF NOT EXISTS idx_sentences_doc ON sentences(doc_id);
                CREATE INDEX IF NOT EXISTS idx_sentences_clause ON sentences(clause_id);
                CREATE INDEX IF NOT EXISTS idx_sentences_label ON sentences(label);
            """)
        logger.debug(f"Registry initialized: {self.db_path}")

    # ─── Document operations ─────────────────────────────────────────────────

    def upsert_document(self, doc: dict) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO documents (
                    doc_id, filename, url, submitter, category,
                    sha256, size_bytes, num_pages, word_count,
                    downloaded, extracted, created_at, updated_at
                ) VALUES (
                    :doc_id, :filename, :url, :submitter, :category,
                    :sha256, :size_bytes, :num_pages, :word_count,
                    :downloaded, :extracted, :created_at, :updated_at
                )
                ON CONFLICT(doc_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    extracted  = excluded.extracted,
                    word_count = excluded.word_count,
                    num_pages  = excluded.num_pages
            """, {
                "doc_id": doc.get("doc_id", doc.get("filename", "")[:12]),
                "filename": doc.get("filename", ""),
                "url": doc.get("url", ""),
                "submitter": doc.get("submitter", "Unknown"),
                "category": doc.get("category", "individual"),
                "sha256": doc.get("sha256", ""),
                "size_bytes": doc.get("size_bytes", 0),
                "num_pages": doc.get("num_pages", 0),
                "word_count": doc.get("word_count", 0),
                "downloaded": 1,
                "extracted": 1 if doc.get("extracted") else 0,
                "created_at": now,
                "updated_at": now,
            })

    def mark_stage(self, doc_id: str, stage: str, value: bool = True) -> None:
        """Mark a processing stage as complete."""
        stages = {"downloaded", "extracted", "cleaned", "linked", "labeled"}
        if stage not in stages:
            raise ValueError(f"Unknown stage: {stage}. Must be one of {stages}")
        with self._conn() as conn:
            conn.execute(
                f"UPDATE documents SET {stage} = ?, updated_at = ? WHERE doc_id = ?",
                (1 if value else 0, datetime.utcnow().isoformat(), doc_id),
            )

    def get_document(self, doc_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def get_all_documents(self, stage_filter: Optional[str] = None) -> list[dict]:
        with self._conn() as conn:
            if stage_filter:
                rows = conn.execute(
                    f"SELECT * FROM documents WHERE {stage_filter} = 1"
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM documents").fetchall()
        return [dict(r) for r in rows]

    # ─── Sentence operations ─────────────────────────────────────────────────

    def insert_sentences(self, sentences: list[dict]) -> None:
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO sentences (
                    sent_id, doc_id, clause_id, text, language,
                    translated_text, similarity_score, label, confidence, label_source
                ) VALUES (
                    :sent_id, :doc_id, :clause_id, :text, :language,
                    :translated_text, :similarity_score, :label, :confidence, :label_source
                )
            """, sentences)

    def get_sentences(
        self,
        doc_id: Optional[str] = None,
        clause_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> list[dict]:
        query = "SELECT * FROM sentences WHERE 1=1"
        params = []
        if doc_id:
            query += " AND doc_id = ?"
            params.append(doc_id)
        if clause_id:
            query += " AND clause_id = ?"
            params.append(clause_id)
        if label:
            query += " AND label = ?"
            params.append(label)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self._conn() as conn:
            docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            sents = conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
            labeled = conn.execute("SELECT COUNT(*) FROM sentences WHERE label IS NOT NULL").fetchone()[0]
            label_dist = conn.execute(
                "SELECT label, COUNT(*) as cnt FROM sentences WHERE label IS NOT NULL GROUP BY label"
            ).fetchall()
        return {
            "total_documents": docs,
            "total_sentences": sents,
            "labeled_sentences": labeled,
            "label_distribution": {row[0]: row[1] for row in label_dist},
        }

    def export_jsonl(self, output_path: str, labeled_only: bool = True) -> int:
        """Export sentences as JSONL for training."""
        sentences = self.get_sentences()
        if labeled_only:
            sentences = [s for s in sentences if s.get("label")]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"Exported {len(sentences)} sentences → {output_path}")
        return len(sentences)
