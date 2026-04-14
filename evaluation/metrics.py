"""
IndiaLexABSA — Evaluation Metrics
=====================================
Computes comprehensive evaluation metrics for all models:
  - Accuracy, macro-F1, weighted-F1
  - Per-class precision/recall/F1
  - Confusion matrix
  - Expected Calibration Error (ECE)
  - Per-clause accuracy breakdown
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from loguru import logger

LABELS = ["supportive", "critical", "suggestive", "neutral", "ambiguous"]


def compute_all_metrics(
    y_true: list[str],
    y_pred: list[str],
    probs: Optional[np.ndarray] = None,
    model_name: str = "model",
) -> dict:
    """
    Compute comprehensive metrics for a classifier.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        probs: Probability matrix (N x num_classes) for calibration metrics
        model_name: Name for logging
    
    Returns: Dict of all metrics
    """
    # Convert to indices for sklearn
    label2id = {l: i for i, l in enumerate(LABELS)}
    y_true_int = [label2id.get(l, 3) for l in y_true]
    y_pred_int = [label2id.get(l, 3) for l in y_pred]

    acc = accuracy_score(y_true_int, y_pred_int)
    macro_f1 = f1_score(y_true_int, y_pred_int, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_int, y_pred_int, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_true_int, y_pred_int, average=None, zero_division=0, labels=list(range(len(LABELS))))
    per_class_prec = precision_score(y_true_int, y_pred_int, average=None, zero_division=0, labels=list(range(len(LABELS))))
    per_class_rec = recall_score(y_true_int, y_pred_int, average=None, zero_division=0, labels=list(range(len(LABELS))))
    cm = confusion_matrix(y_true_int, y_pred_int, labels=list(range(len(LABELS))))

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            LABELS[i]: {
                "f1": round(float(per_class_f1[i]), 4),
                "precision": round(float(per_class_prec[i]), 4),
                "recall": round(float(per_class_rec[i]), 4),
                "support": int(y_true_int.count(i)),
            }
            for i in range(len(LABELS))
        },
        "confusion_matrix": cm.tolist(),
    }

    # Expected Calibration Error (ECE)
    if probs is not None:
        metrics["ece"] = round(_compute_ece(y_true_int, probs), 4)

    logger.info(
        f"[{model_name}] Acc={acc:.3f} | Macro-F1={macro_f1:.3f} | Weighted-F1={weighted_f1:.3f}"
    )
    return metrics


def _compute_ece(y_true: list[int], probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    max_probs = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == np.array(y_true)).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (max_probs > bin_edges[i]) & (max_probs <= bin_edges[i + 1])
        if mask.sum() > 0:
            avg_conf = max_probs[mask].mean()
            avg_acc = correct[mask].mean()
            ece += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / len(y_true))


def per_clause_metrics(
    y_true: list[str],
    y_pred: list[str],
    clause_ids: list[str],
) -> pd.DataFrame:
    """Compute accuracy and F1 breakdown per clause."""
    records = []
    unique_clauses = sorted(set(clause_ids))
    for clause in unique_clauses:
        mask = [i for i, c in enumerate(clause_ids) if c == clause]
        if not mask:
            continue
        ct = [y_true[i] for i in mask]
        cp = [y_pred[i] for i in mask]
        acc = accuracy_score(ct, cp)
        f1 = f1_score(ct, cp, average="macro", zero_division=0)
        records.append({"clause_id": clause, "n": len(mask), "accuracy": acc, "macro_f1": f1})
    return pd.DataFrame(records).sort_values("n", ascending=False)


def metrics_to_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """Convert list of model metrics dicts to a comparison DataFrame."""
    rows = []
    for m in metrics_list:
        row = {
            "Model": m["model"],
            "Accuracy": m["accuracy"],
            "Macro-F1": m["macro_f1"],
            "Weighted-F1": m["weighted_f1"],
        }
        for label in LABELS:
            row[f"F1-{label.capitalize()[:4]}"] = m["per_class"].get(label, {}).get("f1", 0.0)
        if "ece" in m:
            row["ECE"] = m["ece"]
        rows.append(row)
    return pd.DataFrame(rows)


def print_classification_report(y_true: list[str], y_pred: list[str]) -> str:
    label2id = {l: i for i, l in enumerate(LABELS)}
    yt = [label2id.get(l, 3) for l in y_true]
    yp = [label2id.get(l, 3) for l in y_pred]
    return classification_report(yt, yp, target_names=LABELS, zero_division=0)
