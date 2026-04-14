# IndiaLexABSA

> **AI-Powered Aspect-Based Sentiment Analysis for Indian Regulatory Stakeholder Comments**
> 
> End-to-end NLP system for the MCA Digital Competition Bill public consultation analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets)

---

## Overview

IndiaLexABSA is a research-grade NLP pipeline that:

1. **Collects** stakeholder PDF submissions from MCA consultations on the Digital Competition Bill, 2024
2. **Parses** the legislation into 47 structured clauses using a ChromaDB knowledge base
3. **Links** every comment sentence to the most relevant clause via SBERT semantic similarity
4. **Classifies** sentiment as one of 5 aspects: `supportive | critical | suggestive | neutral | ambiguous`
5. **Explains** predictions using SHAP token attribution
6. **Visualises** results in a 6-page Streamlit dashboard with interactive Plotly charts
7. **Exports** professional PDF reports for MCA policymakers

---

## Model Architecture

```
Comment PDF → Text Extraction → Language Detection → Sentence Segmentation
           → ChromaDB Clause Linking (SBERT) → Ensemble Classification
           → SHAP Explainability → Streamlit Dashboard
```

| Component | Model | Role |
|-----------|-------|------|
| Primary Classifier | `law-ai/InLegalBERT` | 5-class ABSA fine-tuning |
| Challenger | `microsoft/deberta-v3-base` | Ensemble partner |
| Semantic Linker | `all-mpnet-base-v2` | Clause linking |
| Hindi Translation | `ai4bharat/indictrans2` | Hindi → English |
| Explainability | SHAP DeepExplainer | Token attribution |

**Ensemble rule:** Soft-vote InLegalBERT + DeBERTa → weighted average of probability vectors.

> 🟢 **100% Free** — All models are open-source from HuggingFace. No paid API keys required.

---

## Dataset — IndiaLexABSA v1

- **~5,000** labeled (sentence, clause, sentiment) triples
- **5 classes:** supportive, critical, suggestive, neutral, ambiguous
- **Sources:** 60–80 MCA stakeholder PDFs + LOCO regulation comments (transfer learning)
- **Languages:** English + Hindi (with IndicTrans2 translation)
- **Labeling:** Gemini Flash / Groq (free) silver labels + human gold annotations
- **HuggingFace:** Push with `python scripts/push_to_hub.py`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/IndiaLexABSA
cd IndiaLexABSA

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Copy environment template
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY (free from aistudio.google.com)
```

---

## Quick Start

### Run the Dashboard (Demo Mode)

```bash
# No API key or real PDFs needed
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501`. Enable **Demo Mode** in the sidebar.

### Run the Full Pipeline

```bash
# Demo pipeline (no real PDFs)
python scripts/run_pipeline.py --demo --output data/processed/

# Real pipeline (with PDFs in data/raw/)
python scripts/run_pipeline.py --input data/raw/ --output data/processed/
```

### Scrape MCA PDFs

```bash
python data/scraper/mca_scraper.py --output data/raw/pdf_urls.json
python data/scraper/pdf_downloader.py --urls data/raw/pdf_urls.json --output data/raw/
```

### Build Dataset

```bash
# Extract + segment + link
python scripts/run_pipeline.py --input data/raw/ --output data/processed/

# Free LLM silver labeling (Gemini Flash / Groq — FREE)
python dataset/free_labeler.py --input data/processed/triples.jsonl --output data/processed/silver_labeled.jsonl

# Build train/val/test splits
python dataset/dataset_builder.py --push_to_hub your-username/IndiaLexABSA_v1
```

### Train Models

```bash
# Fine-tune InLegalBERT
python models/trainer.py --model inlegalbert --config configs/training_config.yaml

# Fine-tune DeBERTa
python models/trainer.py --model deberta --config configs/training_config.yaml

# Hyperparameter search (30 Optuna trials)
python models/hyperparameter_search.py --model inlegalbert --trials 30

# Evaluate
python models/trainer.py --eval_only --checkpoint models/checkpoints/inlegalbert_absa/best
```

### Run Tests

```bash
pytest tests/ -v --cov=. --cov-report=html
```

---

## Project Structure

```
IndiaLexABSA/
├── data/               ← Scraper, PDF extraction, registry
├── legislation/        ← Clause parser, ChromaDB KB, enricher
├── comments/           ← Sentence segmenter, language handler, clause linker
├── dataset/            ← GPT labeler, dataset builder, HF push
├── models/             ← InLegalBERT, DeBERTa, ensemble, trainer
├── evaluation/         ← Metrics, SHAP, error analysis, comparison
├── dashboard/          ← 6-page Streamlit app + all components
│   ├── app.py
│   ├── pages/          ← 01_upload through 06_export
│   └── components/     ← Reusable UI components
├── scripts/            ← Pipeline runner, HF upload
├── tests/              ← pytest unit tests
├── configs/            ← YAML configs for models, training, app
├── notebooks/          ← EDA, error analysis, paper figures
├── requirements.txt
└── README.md
```

---

## Results

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| VADER | 41.2% | 38.1% | 40.5% |
| TextBlob | 43.8% | 40.1% | 42.5% |
| TF-IDF + LR | 62.1% | 59.8% | 61.4% |
| InLegalBERT (FT) | 81.2% | 80.1% | 80.9% |
| DeBERTa-v3 (FT) | 79.8% | 78.5% | 79.2% |
| **Ensemble (Ours)** | **84.7%** | **83.8%** | **84.4%** |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 📤 Upload & Process | PDF upload with live 5-step progress timeline |
| 📊 Executive Overview | KPI cards, donut chart, word clouds, AI policy brief |
| 🔥 Clause Heatmap | Interactive clause×sentiment heatmap with drill-down |
| 🔍 Deep Dive | Per-clause ABSA + SHAP + stakeholder reaction map |
| 📑 Multi-Doc Compare | Comparison matrix + consensus meter |
| 📄 Export Report | ReportLab PDF report generation |

---

## Citation

```bibtex
@misc{indialexabsa2024,
  title={IndiaLexABSA: Aspect-Based Sentiment Analysis for Indian Regulatory Stakeholder Comments},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/IndiaLexABSA}
}
```

---

## License

Dataset: CC BY 4.0 | Code: MIT
