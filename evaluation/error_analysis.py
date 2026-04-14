"""
IndiaLexABSA — Error Analysis
================================
Clusters misclassified examples and identifies failure modes.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


def analyse_errors(
    y_true: list[str],
    y_pred: list[str],
    sentences: list[str],
    clause_ids: Optional[list[str]] = None,
) -> dict:
    """Identify misclassified examples and compute error statistics."""
    errors = [
        {
            "idx": i, "true": y_true[i], "pred": y_pred[i],
            "sentence": sentences[i],
            "clause_id": clause_ids[i] if clause_ids else None,
        }
        for i in range(len(y_true)) if y_true[i] != y_pred[i]
    ]

    # Confusion pairs
    confusion_pairs = Counter((e["true"], e["pred"]) for e in errors)
    top_confusions = confusion_pairs.most_common(10)

    # Error rate by label
    label_total = Counter(y_true)
    label_errors = Counter(e["true"] for e in errors)
    error_rates = {
        label: label_errors.get(label, 0) / max(label_total.get(label, 1), 1)
        for label in set(y_true)
    }

    # Average sentence length for errors vs correct
    correct_lengths = [len(sentences[i].split()) for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    error_lengths = [len(e["sentence"].split()) for e in errors]

    result = {
        "total_errors": len(errors),
        "error_rate": len(errors) / max(len(y_true), 1),
        "top_confusions": [{"true": p[0], "pred": p[1], "count": c} for p, c in top_confusions],
        "error_rate_by_label": error_rates,
        "avg_length_correct": np.mean(correct_lengths) if correct_lengths else 0,
        "avg_length_error": np.mean(error_lengths) if error_lengths else 0,
        "sample_errors": errors[:20],
    }
    logger.info(f"Error analysis: {len(errors)} errors ({result['error_rate']:.1%} error rate)")
    return result


def cluster_errors(errors: list[dict], n_clusters: int = 5) -> list[dict]:
    """Cluster misclassified examples using KMeans on SBERT embeddings."""
    if len(errors) < n_clusters:
        return errors

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        texts = [e["sentence"] for e in errors]
        embeddings = encoder.encode(texts, show_progress_bar=False)

        kmeans = KMeans(n_clusters=min(n_clusters, len(errors)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        for e, cluster_id in zip(errors, labels):
            e["cluster"] = int(cluster_id)

        # Name clusters by most common true-label + pred-label pair
        cluster_map = {}
        for cluster_id in range(n_clusters):
            cluster_errors = [e for e in errors if e.get("cluster") == cluster_id]
            if cluster_errors:
                pairs = Counter((e["true"], e["pred"]) for e in cluster_errors)
                top = pairs.most_common(1)[0][0]
                cluster_map[cluster_id] = f"{top[0]}→{top[1]} ({len(cluster_errors)} errors)"

        for e in errors:
            e["cluster_name"] = cluster_map.get(e.get("cluster"), "Unknown")

    except Exception as exc:
        logger.warning(f"Clustering failed: {exc}")

    return errors
