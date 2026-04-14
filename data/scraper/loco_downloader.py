"""
IndiaLexABSA — LOCO Regulation Comments Downloader
===================================================
Downloads the LOCO (Legislation and Regulation Comments) dataset
for transfer learning. LOCO contains 3.5M US regulation comments
labeled positive/negative/neutral — used for step-1 pretraining
before fine-tuning on IndiaLexABSA 5-class data.

Dataset: https://huggingface.co/datasets/coastalcph/loco
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from loguru import logger


def download_loco(output_dir: str, max_samples: int = 50_000) -> None:
    """Download and preprocess LOCO dataset for transfer learning."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        logger.info("Downloading LOCO dataset from HuggingFace...")
        ds = load_dataset("coastalcph/loco", split="train", streaming=True)

        count = 0
        out_path = out / "loco_3class.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for example in ds:
                if count >= max_samples:
                    break
                # Map LOCO labels to simplified 3-class
                raw_label = example.get("label", 1)
                if raw_label == 0:
                    mapped = "critical"      # negative → critical
                elif raw_label == 2:
                    mapped = "supportive"    # positive → supportive
                else:
                    mapped = "neutral"

                record = {
                    "sent_id": f"loco_{count:07d}",
                    "sentence": example.get("text", "")[:512],
                    "clause_id": example.get("regulation_id", ""),
                    "clause_title": example.get("title", ""),
                    "clause_text": "",
                    "label": mapped,
                    "confidence": 5,
                    "label_source": "loco_dataset",
                    "category": "individual",
                    "submitter": "LOCO",
                    "language": "en",
                }
                f.write(json.dumps(record) + "\n")
                count += 1

                if count % 5000 == 0:
                    logger.info(f"Downloaded {count:,}/{max_samples:,} LOCO samples")

        logger.info(f"LOCO download complete: {count:,} samples → {out_path}")

    except Exception as exc:
        logger.error(f"LOCO download failed: {exc}")
        logger.info("Generating LOCO stubs for testing...")
        _generate_loco_stubs(out, n=1000)


def _generate_loco_stubs(out: Path, n: int = 1000) -> None:
    """Generate stub LOCO data for testing without internet access."""
    import random
    random.seed(42)
    templates = {
        "critical": [
            "This regulation imposes excessive compliance costs on small businesses.",
            "We strongly oppose the proposed rule as it lacks economic justification.",
            "The agency has failed to consider less burdensome alternatives.",
        ],
        "supportive": [
            "We support this regulation as it will protect consumers effectively.",
            "The proposed rule is well-designed and serves the public interest.",
            "We commend the agency for this thoughtful regulatory approach.",
        ],
        "neutral": [
            "The regulation as proposed would take effect on January 1st.",
            "Section 3 provides for compliance reporting on a quarterly basis.",
            "The agency received 1,247 comments on the proposed rulemaking.",
        ],
    }
    LABELS = ["critical", "supportive", "neutral"]
    out_path = out / "loco_3class.jsonl"
    with open(out_path, "w") as f:
        for i in range(n):
            label = random.choice(LABELS)
            text = random.choice(templates[label])
            record = {
                "sent_id": f"loco_stub_{i:05d}",
                "sentence": text,
                "clause_id": f"REG_{random.randint(1, 100)}",
                "clause_title": "US Regulation Comment",
                "label": label,
                "confidence": 5,
                "label_source": "loco_stub",
                "category": "individual",
                "submitter": "LOCO_stub",
                "language": "en",
            }
            f.write(json.dumps(record) + "\n")
    logger.info(f"Generated {n} LOCO stubs → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/loco/")
    parser.add_argument("--max_samples", type=int, default=50_000)
    args = parser.parse_args()
    download_loco(args.output, args.max_samples)
