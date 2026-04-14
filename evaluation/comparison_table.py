"""
IndiaLexABSA — Comparison Table Generator
==========================================
Generates LaTeX-formatted model comparison tables for the paper
and Plotly bar chart versions for the dashboard.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from evaluation.metrics import metrics_to_dataframe

SENTIMENT_COLORS = {
    "supportive": "#2EC4B6",
    "critical": "#E63946",
    "suggestive": "#F4A261",
    "neutral": "#8D99AE",
    "ambiguous": "#9B5DE5",
}


def generate_latex_table(metrics_list: list[dict]) -> str:
    """Generate LaTeX comparison table for paper."""
    df = metrics_to_dataframe(metrics_list)
    cols = ["Model", "Accuracy", "Macro-F1", "Weighted-F1"]
    df_display = df[cols].copy()
    # Bold best value in each column
    for col in ["Accuracy", "Macro-F1", "Weighted-F1"]:
        best_idx = df_display[col].idxmax()
        df_display.loc[best_idx, col] = f"\\textbf{{{df_display.loc[best_idx, col]:.4f}}}"

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Model Comparison on IndiaLexABSA Test Set}",
        "\\label{tab:model_comparison}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Macro-F1} & \\textbf{Weighted-F1} \\\\",
        "\\hline",
    ]
    for _, row in df_display.iterrows():
        lines.append(f"{row['Model']} & {row['Accuracy']} & {row['Macro-F1']} & {row['Weighted-F1']} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def generate_comparison_chart(metrics_list: list[dict]) -> go.Figure:
    """Generate interactive Plotly comparison bar chart."""
    df = metrics_to_dataframe(metrics_list)
    models = df["Model"].tolist()
    metrics = ["Accuracy", "Macro-F1", "Weighted-F1"]
    colors = ["#2EC4B6", "#E63946", "#F4A261"]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=df[metric].tolist(),
            marker_color=color,
            text=[f"{v:.3f}" for v in df[metric].tolist()],
            textposition="outside",
        ))

    fig.update_layout(
        title="Model Performance Comparison — IndiaLexABSA",
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.05]),
        xaxis_title="Model",
        legend_title="Metric",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=13),
        height=500,
    )
    return fig


def get_demo_metrics() -> list[dict]:
    """Return realistic demo metrics for all 6 models."""
    return [
        {"model": "VADER", "accuracy": 0.412, "macro_f1": 0.381, "weighted_f1": 0.405,
         "per_class": {l: {"f1": 0.35, "precision": 0.36, "recall": 0.34, "support": 100} for l in ["supportive","critical","suggestive","neutral","ambiguous"]}},
        {"model": "TextBlob", "accuracy": 0.438, "macro_f1": 0.401, "weighted_f1": 0.425,
         "per_class": {l: {"f1": 0.38, "precision": 0.39, "recall": 0.37, "support": 100} for l in ["supportive","critical","suggestive","neutral","ambiguous"]}},
        {"model": "TF-IDF + LR", "accuracy": 0.621, "macro_f1": 0.598, "weighted_f1": 0.614,
         "per_class": {l: {"f1": 0.57, "precision": 0.59, "recall": 0.55, "support": 100} for l in ["supportive","critical","suggestive","neutral","ambiguous"]}},
        {"model": "GPT-4o Few-Shot", "accuracy": 0.743, "macro_f1": 0.729, "weighted_f1": 0.738,
         "per_class": {l: {"f1": 0.71, "precision": 0.73, "recall": 0.69, "support": 100} for l in ["supportive","critical","suggestive","neutral","ambiguous"]}},
        {"model": "InLegalBERT (FT)", "accuracy": 0.812, "macro_f1": 0.801, "weighted_f1": 0.809,
         "per_class": {"supportive": {"f1": 0.84, "precision": 0.86, "recall": 0.82, "support": 180},
                       "critical": {"f1": 0.83, "precision": 0.85, "recall": 0.81, "support": 220},
                       "suggestive": {"f1": 0.78, "precision": 0.80, "recall": 0.76, "support": 130},
                       "neutral": {"f1": 0.82, "precision": 0.84, "recall": 0.80, "support": 310},
                       "ambiguous": {"f1": 0.68, "precision": 0.70, "recall": 0.66, "support": 80}}},
        {"model": "Ensemble (Ours)", "accuracy": 0.847, "macro_f1": 0.838, "weighted_f1": 0.844,
         "per_class": {"supportive": {"f1": 0.87, "precision": 0.89, "recall": 0.85, "support": 180},
                       "critical": {"f1": 0.86, "precision": 0.88, "recall": 0.84, "support": 220},
                       "suggestive": {"f1": 0.82, "precision": 0.84, "recall": 0.80, "support": 130},
                       "neutral": {"f1": 0.86, "precision": 0.87, "recall": 0.85, "support": 310},
                       "ambiguous": {"f1": 0.73, "precision": 0.75, "recall": 0.71, "support": 80}}},
    ]
