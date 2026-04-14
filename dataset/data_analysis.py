"""
IndiaLexABSA — Dataset Analysis
=================================
Class distribution, agreement stats, sentence length histograms,
per-clause coverage report and label quality metrics.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


LABELS = ["supportive", "critical", "suggestive", "neutral", "ambiguous"]
COLORS = ["#2EC4B6", "#E63946", "#F4A261", "#8D99AE", "#9B5DE5"]


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def class_distribution(items: list[dict]) -> dict:
    counts = Counter(item.get("label","neutral") for item in items)
    total = sum(counts.values())
    return {
        "counts": dict(counts),
        "percentages": {k: v/total*100 for k, v in counts.items()},
        "total": total,
    }


def sentence_length_stats(items: list[dict]) -> dict:
    lengths = [len(item.get("sentence", item.get("text","")).split()) for item in items]
    return {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
    }


def per_clause_coverage(items: list[dict]) -> pd.DataFrame:
    clause_map: dict[str, Counter] = {}
    for item in items:
        cid = item.get("clause_id","")
        lbl = item.get("label","neutral")
        if cid:
            clause_map.setdefault(cid, Counter())[lbl] += 1

    rows = []
    for cid, cnts in sorted(clause_map.items(), key=lambda x: sum(x[1].values()), reverse=True):
        total = sum(cnts.values())
        dominant = max(cnts, key=cnts.get) if cnts else "neutral"
        rows.append({
            "clause_id": cid,
            "total_comments": total,
            "dominant_label": dominant,
            **{f"n_{l}": cnts.get(l, 0) for l in LABELS},
        })
    return pd.DataFrame(rows)


def label_source_analysis(items: list[dict]) -> dict:
    source_counts = Counter(item.get("label_source","unknown") for item in items)
    return dict(source_counts)


def plot_class_distribution(items: list[dict], output_path: str = "") -> plt.Figure:
    dist = class_distribution(items)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    labels_present = [l for l in LABELS if l in dist["counts"]]
    counts = [dist["counts"].get(l, 0) for l in labels_present]
    colors_present = [COLORS[LABELS.index(l)] for l in labels_present]

    axes[0].bar(
        [l.capitalize() for l in labels_present], counts,
        color=colors_present, edgecolor="white", linewidth=0.5,
    )
    axes[0].set_title("Label Distribution (Count)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].spines[["top","right"]].set_visible(False)
    for i, v in enumerate(counts):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=9)

    # Pie chart
    axes[1].pie(
        [dist["counts"].get(l, 0) for l in LABELS],
        labels=[l.capitalize() for l in LABELS],
        colors=COLORS,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.82,
    )
    axes[1].set_title("Label Distribution (%)", fontsize=12, fontweight="bold")

    plt.suptitle(f"IndiaLexABSA Dataset — {dist['total']} samples",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot → {output_path}")
    return fig


def plot_length_distribution(items: list[dict], output_path: str = "") -> plt.Figure:
    lengths_by_label = {l: [] for l in LABELS}
    for item in items:
        lbl = item.get("label","neutral")
        text = item.get("sentence", item.get("text",""))
        if lbl in LABELS:
            lengths_by_label[lbl].append(len(text.split()))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(LABELS):
        if lengths_by_label[label]:
            ax.hist(lengths_by_label[label], bins=30, alpha=0.6,
                    label=label.capitalize(), color=COLORS[i])
    ax.set_xlabel("Sentence Length (words)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Sentence Length Distribution by Label", fontsize=12, fontweight="bold")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def full_report(dataset_dir: str = "dataset/IndiaLexABSA_v1") -> dict:
    """Run full analysis on all splits."""
    splits = {}
    for split in ["train", "validation", "test"]:
        path = Path(dataset_dir) / f"{split}.jsonl"
        if path.exists():
            items = load_jsonl(str(path))
            splits[split] = {
                "n": len(items),
                "distribution": class_distribution(items),
                "length_stats": sentence_length_stats(items),
                "source_stats": label_source_analysis(items),
            }
            logger.info(f"{split}: {len(items)} items, {dict(class_distribution(items)['counts'])}")

    return splits


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset/IndiaLexABSA_v1")
    parser.add_argument("--output_dir", default="notebooks/figures/")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    report = full_report(args.dataset_dir)
    print(json.dumps(report, indent=2, default=str))
