# IndiaLexABSA — Full Professional Implementation Plan

## Project Overview

**IndiaLexABSA** is an end-to-end Aspect-Based Sentiment Analysis (ABSA) system for Indian government regulatory stakeholder comments. It processes MCA (Ministry of Corporate Affairs) PDF submissions on the Digital Competition Bill, links comment sentences to law clauses, classifies sentiment using fine-tuned transformer ensemble, and presents results through a professional Streamlit dashboard.

This plan reflects the current state of NLP (2024–2025): updated model availability, modern tooling, and best practices.

---

## Architecture at a Glance

```
PDF Submissions → Text Extraction → Language Detection → Clause Linking
              → Sentiment Classification (Ensemble) → ChromaDB Storage
              → Streamlit Dashboard (6 pages) → PDF Report
```

**5-class sentiment**: `supportive` | `critical` | `suggestive` | `neutral` | `ambiguous`

---

## Model Stack (Updated)

| Role | Model | Why |
|------|-------|-----|
| Primary classifier | `law-ai/InLegalBERT` | Pre-trained on Indian SC/HC judgments |
| Challenger | `microsoft/deberta-v3-base` | Disentangled attention for NLI-style legal text |
| Semantic linker | `sentence-transformers/all-mpnet-base-v2` | Cosine-sim clause linking (not fine-tuned) |
| Hindi translation | `ai4bharat/indictrans2-en-indic-dist-200M` | Best open-source Hindi↔English translator |
| Hindi aux classifier | `ai4bharat/IndicBERT` | Fine-tuned on IndicSentiment |
| LLM backbone | `gpt-4o-mini` (bulk) + `gpt-4o` (reports) | Labeling + summaries + executive brief |
| Explainability | SHAP + `captum` | Token-level attribution |

**Ensemble strategy**: Soft-vote InLegalBERT + DeBERTa → if both confidence < 0.65 → GPT-4o few-shot fallback (~8–12% of sentences)

---

## Key Design Decisions

> [!IMPORTANT]
> **ChromaDB** replaces a traditional SQL clause store. Clause embeddings live in ChromaDB for fast semantic search. This is the modern RAG-style approach.

> [!IMPORTANT]
> **HuggingFace Datasets** — the final labeled dataset is pushed to HuggingFace Hub as `IndiaLexABSA_v1` with train/val/test splits. This is a core deliverable.

> [!NOTE]
> All expensive GPU training runs are designed for **Google Colab** / local GPU. CPU inference path is available for the dashboard demo mode (loads quantized models).

> [!WARNING]
> OpenAI API calls require an `OPENAI_API_KEY` environment variable. The system gracefully degrades: if no key is set, GPT features show placeholder text and the silver-labeling step is skipped.

---

## Proposed File Structure

```
IndiaLexABSA/
│
├── data/
│   ├── scraper/
│   │   ├── mca_scraper.py          ← Selenium scraper for MCA portal
│   │   ├── pdf_downloader.py       ← Bulk PDF downloader with dedup
│   │   └── loco_downloader.py      ← Downloads LOCO regulation comments
│   ├── ingestion/
│   │   ├── pdf_extractor.py        ← PyMuPDF + pdfplumber hybrid extraction
│   │   ├── text_cleaner.py         ← Legal text normalization pipeline
│   │   └── data_registry.py        ← JSON registry of all processed docs
│   ├── raw/                        ← Downloaded PDFs [git-ignored]
│   └── processed/                  ← Extracted texts as JSONL
│
├── legislation/
│   ├── clause_parser.py            ← Regex + spaCy clause boundary detection
│   ├── clause_enricher.py          ← GPT-4o plain-English clause summaries
│   ├── knowledge_base.py           ← ChromaDB collection manager
│   └── cross_referencer.py         ← Inter-clause reference graph
│
├── comments/
│   ├── sentence_segmenter.py       ← spaCy + NLTK hybrid sentence splitter
│   ├── language_handler.py         ← langdetect + IndicTrans2 translation
│   ├── clause_linker.py            ← SBERT cosine-sim + MMR re-ranking
│   └── context_builder.py          ← Builds (sentence, clause, context) triples
│
├── dataset/
│   ├── gpt_labeler.py              ← GPT-4o-mini silver labeling with CoT
│   ├── human_annotation_tool.py    ← Streamlit-based annotation interface
│   ├── dataset_builder.py          ← Merges silver + human labels, HF push
│   ├── data_analysis.py            ← Class distribution, agreement stats
│   └── IndiaLexABSA_v1/           ← HuggingFace dataset (train/val/test)
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── models/
│   ├── baseline_models.py          ← TF-IDF+LR, TextBlob, VADER baselines
│   ├── inlegalbert_absa.py         ← InLegalBERT fine-tuning (HF Trainer)
│   ├── deberta_absa.py             ← DeBERTa-v3 fine-tuning
│   ├── llm_few_shot.py             ← GPT-4o few-shot with structured output
│   ├── ensemble.py                 ← Soft-vote + confidence-based GPT fallback
│   ├── trainer.py                  ← Unified training entrypoint + WandB logging
│   └── hyperparameter_search.py    ← Optuna HPO for both models
│
├── evaluation/
│   ├── metrics.py                  ← Per-class F1, confusion matrix, ECE
│   ├── error_analysis.py           ← Failure mode clustering
│   ├── explainability.py           ← SHAP + Integrated Gradients
│   └── comparison_table.py         ← LaTeX-ready comparison table generator
│
├── dashboard/
│   ├── app.py                      ← Main Streamlit entry, sidebar nav
│   ├── pages/
│   │   ├── 01_upload.py            ← Upload + live pipeline progress
│   │   ├── 02_overview.py          ← KPI cards + charts + AI brief
│   │   ├── 03_heatmap.py           ← Interactive clause×sentiment heatmap
│   │   ├── 04_deep_dive.py         ← Per-clause ABSA + SHAP
│   │   ├── 05_comparison.py        ← Multi-doc comparison matrix
│   │   └── 06_export.py            ← PDF report generation
│   └── components/
│       ├── sentiment_badge.py      ← Colored sentiment pill component
│       ├── clause_card.py          ← Clause display with highlights
│       ├── kpi_card.py             ← Metric KPI card component
│       ├── word_cloud_gen.py       ← Per-sentiment word cloud generator
│       └── pdf_report.py           ← ReportLab PDF builder
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← Exploratory data analysis
│   ├── 02_error_analysis.ipynb     ← Failure mode deep dive
│   └── 03_paper_figures.ipynb      ← Publication-quality figures
│
├── configs/
│   ├── model_config.yaml           ← Model paths, hyperparameters
│   ├── training_config.yaml        ← Training loop settings
│   └── app_config.yaml             ← Dashboard settings
│
├── tests/
│   ├── test_pdf_extractor.py
│   ├── test_clause_linker.py
│   ├── test_ensemble.py
│   └── test_dashboard_components.py
│
├── scripts/
│   ├── run_pipeline.py             ← End-to-end pipeline runner
│   ├── pretrain_domain_adapt.py    ← Masked LM domain adaptation
│   └── push_to_hub.py              ← HuggingFace Hub upload script
│
├── .env.example                    ← Template for API keys
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

---

## Proposed Changes — Component by Component

### Infrastructure & Config

#### [NEW] `requirements.txt`
All dependencies pinned: transformers, datasets, sentence-transformers, chromadb, streamlit, plotly, reportlab, shap, spacy, pdfplumber, pymupdf, langdetect, optuna, wandb, openai, wordcloud, indictrans.

#### [NEW] `configs/model_config.yaml`
Model IDs, checkpoint paths, ChromaDB collection names, confidence thresholds.

#### [NEW] `configs/training_config.yaml`
Batch size, learning rate, warmup steps, max epochs, early stopping patience.

#### [NEW] `configs/app_config.yaml`
Streamlit theme, demo mode flag, API timeout settings.

#### [NEW] `.env.example`
```
OPENAI_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here
WANDB_API_KEY=your_key_here
```

#### [NEW] `.gitignore`
Ignores: `data/raw/`, `*.pdf`, `__pycache__/`, `.env`, `wandb/`, `*.ckpt`

---

### Data Pipeline

#### [NEW] `data/scraper/mca_scraper.py`
Selenium-based scraper for `https://www.mca.gov.in`. Finds consultation submission links, paginates, collects PDF URLs. Includes rate limiting and retry logic.

#### [NEW] `data/scraper/pdf_downloader.py`
Downloads PDFs with SHA-256 deduplication. Logs each download to `data_registry.json`. Supports resumable downloads.

#### [NEW] `data/ingestion/pdf_extractor.py`
Hybrid extractor: PyMuPDF for digital PDFs, pdfplumber as fallback. Handles multi-column layouts, tables, footnotes. Returns structured JSONL with page numbers.

#### [NEW] `data/ingestion/text_cleaner.py`
Legal text normalization: removes headers/footers, normalizes whitespace, handles ligatures, fixes hyphen breaks, strips page numbers.

#### [NEW] `data/ingestion/data_registry.py`
SQLite-backed registry tracking each PDF (path, SHA-256, extraction status, sentence count, language).

---

### Legislation Module

#### [NEW] `legislation/clause_parser.py`
Regex + spaCy-based parser for Indian legislation structure. Handles:
- Section/Sub-section/Clause/Proviso hierarchy
- Schedule references
- Cross-references ("as defined in Section 3(1)")
Returns structured JSON: `{id, title, text, level, parent_id, cross_refs}`

#### [NEW] `legislation/clause_enricher.py`
GPT-4o call per clause: generates plain-English summary, identifies key legal concepts, flags ambiguous provisions. Results cached to avoid re-calling.

#### [NEW] `legislation/knowledge_base.py`
ChromaDB manager. Stores clause embeddings (via SBERT). Provides `search(query, top_k)` returning ranked clause matches with scores.

#### [NEW] `legislation/cross_referencer.py`
Builds a NetworkX directed graph of inter-clause references. Used for context window expansion in clause linking.

---

### Comments Module

#### [NEW] `comments/sentence_segmenter.py`
spaCy `en_core_web_trf` pipeline with custom legal sentence boundary rules. Handles numbered lists, bullet points, quotations.

#### [NEW] `comments/language_handler.py`
`langdetect` for language identification. IndicTrans2 for Hindi→English translation. Caches translations to avoid re-running.

#### [NEW] `comments/clause_linker.py`
Core linking logic:
1. Embed sentence with SBERT
2. Search ChromaDB for top-5 clause candidates
3. MMR (Maximal Marginal Relevance) re-ranking to avoid duplicate links
4. Confidence threshold filtering (default 0.35)
Returns: `{sentence_id, clause_id, similarity_score, is_translated}`

#### [NEW] `comments/context_builder.py`
Assembles final `(sentence, clause_text, clause_summary, surrounding_context)` training triples. Produces JSONL ready for labeling.

---

### Dataset Module

#### [NEW] `dataset/gpt_labeler.py`
GPT-4o-mini silver labeling with Chain-of-Thought prompting. Structured output via JSON mode. Generates:
- Primary label (5-class)
- Confidence score (1–5)
- One-sentence rationale
Processes in batches of 20, handles rate limits with exponential backoff.

#### [NEW] `dataset/human_annotation_tool.py`
Streamlit app for human annotation. Shows sentence, clause context, GPT label as suggestion. Annotator selects final label. Saves to SQLite with inter-annotator agreement tracking.

#### [NEW] `dataset/dataset_builder.py`
Merges silver labels + human corrections. Applies quality filters (confidence threshold, agreement score). Generates train/val/test splits (70/15/15 stratified). Pushes to HuggingFace Hub.

#### [NEW] `dataset/data_analysis.py`
Class distribution plots, label agreement statistics, inter-annotator kappa, sentence length histograms, per-clause coverage report.

---

### Models

#### [NEW] `models/baseline_models.py`
Three baselines:
1. VADER sentiment (converted to 5-class)
2. TF-IDF + Logistic Regression
3. TextBlob polarity bucketing
All wrapped in consistent `predict(sentences)` interface.

#### [NEW] `models/inlegalbert_absa.py`
HuggingFace Trainer fine-tuning of `law-ai/InLegalBERT`:
- Custom classification head with dropout
- Class-weighted loss for imbalanced sentiment
- Gradient checkpointing for memory efficiency
- WandB logging
- Saves best checkpoint by val macro-F1

#### [NEW] `models/deberta_absa.py`
Same structure as InLegalBERT but for `microsoft/deberta-v3-base`. Uses different tokenizer (SentencePiece-based). Slightly higher learning rate due to model size.

#### [NEW] `models/llm_few_shot.py`
GPT-4o few-shot classifier using 10 curated examples (2 per class). Uses OpenAI structured outputs (JSON mode). Tracks token usage and cost. Only called for low-confidence ensemble cases.

#### [NEW] `models/ensemble.py`
Soft-vote ensemble:
```python
final_prob = 0.5 * inlegalbert_prob + 0.5 * deberta_prob
if max(final_prob) < 0.65:
    final_label = llm_few_shot.predict(sentence)  # GPT fallback
```
Returns prediction + confidence + whether GPT was used.

#### [NEW] `models/trainer.py`
Unified CLI entrypoint: `python trainer.py --model inlegalbert --config configs/training_config.yaml`. Handles domain adaptation pre-training (MLM), two-step fine-tuning (LOCO → IndiaLexABSA), and final evaluation.

#### [NEW] `models/hyperparameter_search.py`
Optuna TPE search over: learning rate (1e-5 to 5e-5), batch size (8/16/32), warmup ratio (0.05–0.2), dropout (0.1–0.4). 30 trials, prune unpromising with MedianPruner.

---

### Evaluation

#### [NEW] `evaluation/metrics.py`
Computes: accuracy, macro-F1, per-class F1/precision/recall, confusion matrix, ECE (Expected Calibration Error), per-clause accuracy breakdown.

#### [NEW] `evaluation/error_analysis.py`
Clusters misclassified examples with KMeans on SBERT embeddings. Labels clusters with GPT-4o to identify failure modes ("ambiguous legal hedging", "Hindi code-switching", etc.).

#### [NEW] `evaluation/explainability.py`
SHAP `DeepExplainer` for token attribution. Generates HTML token highlighting. Integrated Gradients via `captum` as alternative.

#### [NEW] `evaluation/comparison_table.py`
Generates LaTeX-formatted comparison table for paper: all 6 models × 5 metrics. Also generates Plotly bar chart version for dashboard.

---

### Dashboard

#### [NEW] `dashboard/app.py`
Main Streamlit app with custom CSS theming. Sidebar navigation. Session state management. Demo mode with cached results when no API key.

#### [NEW] `dashboard/pages/01_upload.py`
- Large drag-and-drop file uploader
- Live progress timeline (5 steps, each lights up green)
- Live counter: "287 sentences linked to 31 clauses"
- Preview of first 3 linked sentences with sentiment badges

#### [NEW] `dashboard/pages/02_overview.py`
- 4 KPI cards row
- Donut chart (Plotly) — dominant sentiment % in center
- Horizontal bar chart — top 10 clauses by comment volume, colored by dominant sentiment
- Three mini word clouds (critical/supportive/suggestive)
- GPT-4o executive brief (3-paragraph policy brief)

#### [NEW] `dashboard/pages/03_heatmap.py`
- Interactive Plotly heatmap (clauses × sentiments)
- Cell click → side panel with matching sentences + confidence scores
- Filter bar: stakeholder type, confidence threshold slider, language filter
- Real-time filter updates

#### [NEW] `dashboard/pages/04_deep_dive.py`
- Clause selector dropdown
- Left panel: full clause text + spaCy NER highlights + complexity score + GPT summary
- Center panel: mini donut + sentence cards with SHAP token coloring
- Right panel: stakeholder reaction table (type | dominant sentiment | representative quote)

#### [NEW] `dashboard/pages/05_comparison.py`
- Upload 2–5 PDFs simultaneously
- Comparison matrix (clauses × submitters) with colored sentiment icons
- Consensus meter per clause (% agreement)
- High-disagreement clauses flagged amber

#### [NEW] `dashboard/pages/06_export.py`
- Single "Generate PDF Report" button
- ReportLab: cover page, executive summary, clause table, top 5 concerns + quotes, word clouds, full data appendix
- Download button for generated PDF

#### [NEW] `dashboard/components/sentiment_badge.py`
HTML sentiment pill: teal/coral/amber/grey/purple.

#### [NEW] `dashboard/components/clause_card.py`
Styled clause display card with expandable text and confidence bar.

#### [NEW] `dashboard/components/kpi_card.py`
Metric KPI card with delta indicator and sparkline.

#### [NEW] `dashboard/components/word_cloud_gen.py`
Per-sentiment word cloud generator with custom color maps.

#### [NEW] `dashboard/components/pdf_report.py`
ReportLab PDF builder with government report styling.

---

### Supporting Files

#### [NEW] `tests/` — 4 test files
Unit tests for PDF extractor, clause linker, ensemble predictor, and dashboard components.

#### [NEW] `scripts/run_pipeline.py`
E2E pipeline: `python run_pipeline.py --input path/to/pdfs --output data/processed/`

#### [NEW] `notebooks/01_EDA.ipynb`
Pre-seeded with data loading code and key visualization cells.

#### [NEW] `README.md`
Professional documentation: installation, quickstart, model card, dataset card, results table, citation.

---

## Implementation Order (Execution Phase)

1. **Infrastructure** — requirements.txt, configs, .gitignore, .env.example
2. **Data pipeline** — pdf_extractor, text_cleaner, data_registry
3. **Legislation module** — clause_parser, knowledge_base, clause_enricher
4. **Comments module** — sentence_segmenter, language_handler, clause_linker, context_builder
5. **Dataset module** — gpt_labeler, dataset_builder, data_analysis
6. **Models** — baselines, inlegalbert_absa, deberta_absa, llm_few_shot, ensemble, trainer
7. **Evaluation** — metrics, error_analysis, explainability, comparison_table
8. **Dashboard** — app.py, all 6 pages, all components
9. **Scripts + Tests** — run_pipeline, unit tests
10. **Documentation** — README.md, notebooks

---

## Open Questions

> [!IMPORTANT]
> **Do you have an OpenAI API key?** GPT-4o features (executive brief, clause enrichment, silver labeling) are central to the design. Without a key, these degrade to placeholder text. The system will work in "demo mode" but the LLM features won't be live.

> [!IMPORTANT]
> **Do you want actual MCA PDF scraping?** The scraper code will be written, but actually downloading 60–80 PDFs requires running the scraper. Do you want me to also include sample/mock data so the dashboard can be demoed immediately without real PDFs?

> [!NOTE]
> **GPU training**: The model fine-tuning code targets Colab/local GPU. For the dashboard demo, I'll add a "load pre-computed results" path so it runs on CPU with cached predictions (no GPU needed to run the dashboard).

> [!NOTE]
> **SHAP in dashboard**: SHAP explanations are computationally expensive. I'll implement them with a caching layer — first call computes and caches, subsequent calls are instant.

---

## Verification Plan

### After Implementation
1. `pip install -r requirements.txt` — clean install
2. `python scripts/run_pipeline.py --demo` — runs with sample data
3. `streamlit run dashboard/app.py` — dashboard loads on localhost:8501
4. Navigate all 6 pages in browser — verify all charts render
5. `pytest tests/` — all unit tests pass

### Quality Checks
- All imports resolve correctly
- Dashboard runs without API key (demo mode)
- PDF report generates successfully
- ChromaDB creates and queries correctly
