![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/license-MIT-green)

# Lead Scoring — Predictive System for SMBs

Predictive lead scoring system combining rule-based heuristics with ML models
(Logistic Regression + Random Forest) to rank B2B leads by conversion probability.

## Features

- **ML Model**: Logistic Regression with 80%+ AUC
- **Explainability**: SHAP-based per-lead explanations  
- **Action Tiers**: Hot/Warm/Nurture/Suppress with SLA recommendations
- **Portfolio Dashboard**: Interactive Streamlit app
- **Production API**: FastAPI with CSV batch scoring
- **GCP Deployment**: Cloud Run with weekly Cloud Scheduler

## Project Structure

```
lead_scoring/
│
├── data/
│   ├── raw/                  # Original source files — never modified
│   └── processed/            # Feature matrix and cleaned datasets
│
├── notebooks/                # One notebook per stage, numbered sequentially
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_scoring.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_output_layer.ipynb
│
├── src/                      # Importable Python modules
│   ├── features/             # Feature engineering logic
│   ├── models/               # Model training and calibration
│   ├── evaluation/           # Metrics, calibration plots, lift curves
│   └── utils/                # Shared helpers (I/O, logging, config)
│
├── outputs/
│   ├── reports/              # EDA report, model comparison, final scorecard
│   ├── figures/              # Saved charts (referenced in reports)
│   ├── models/               # Serialised model artefacts (.pkl)
│   └── scores/               # Scored lead files per run
│
├── docs/                     # Decisions, methodology, feature dictionary
│   ├── decisions.md
│   └── feature_dictionary.md
│
├── app/                      # NEW: FastAPI application
│   ├── core/
│   │   └── scorer.py         # Core scoring logic
│   └── main.py               # FastAPI endpoints
│
├── dashboard/                # NEW: Streamlit portfolio showcase
│   └── app.py
│
├── scripts/
│   └── train_model.py        # Model training & serialization
│
├── config.py                 # Central config: paths, thresholds, hyperparams
├── cloudbuild.yaml           # GCP CI/CD
├── deploy.sh                 # Deployment script
├── docker-compose.yml        # Local development
├── Dockerfile                # Container config
└── README.md
```

## Stages

| Stage | Notebook | Status |
|-------|----------|--------|
| 1 — EDA & Data Audit | `01_eda.ipynb` | ✅ Complete |
| 2 — Feature Engineering | `02_feature_engineering.ipynb` | ✅ Complete |
| 3 — Rule-Based Baseline | `03_baseline_scoring.ipynb` | ✅ Complete |
| 4 — Predictive Modeling | `04_modeling.ipynb` | ✅ Complete |
| 5 — Output & Action Layer | `05_output_layer.ipynb` | ✅ Complete |

## Quick Start

### Local Development

```bash
# Install dependencies
uv sync

# Train the model
uv run python scripts/train_model.py

# Run FastAPI (http://localhost:8000)
uv run uvicorn app.main:app --reload

# Run Streamlit Dashboard (http://localhost:8501)
uv run streamlit run dashboard/app.py

# Or use Docker Compose
docker-compose up
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Score single lead
curl -X POST http://localhost:8000/score/lead \
  -H "Content-Type: application/json" \
  -d '{"lead_id": "lead_001", "features": {"high_intent_touch_count": 5, "is_decision_maker": 1}}'

# Batch score from CSV
curl -X POST http://localhost:8000/score/batch \
  -F "file=@new_leads.csv"
```

## GCP Deployment (Cloud Run)

### Prerequisites

1. [Install gcloud CLI](https://cloud.google.com/sdk/docs/install)
2. Create a GCP project
3. Enable billing (within free tier limits)

### Deploy

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Run deployment script
./deploy.sh
```

This deploys:
- **API**: `https://lead-scoring-api-xxx-uc.a.run.app`
- **Dashboard**: `https://lead-scoring-dashboard-xxx-uc.a.run.app`

### Setup Weekly Batch Job

```bash
gcloud scheduler jobs create http lead-scoring-weekly \
  --schedule="0 9 * * 1" \
  --uri="YOUR_API_URL/score/batch" \
  --http-method=POST \
  --time-zone="America/New_York"
```



## Model Details

- **Algorithm**: Logistic Regression with StandardScaler
- **Features**: 33 behavioral, firmographic, and engagement signals
- **Performance**: AUC ~0.83, Top-20% captures 50%+ conversions
- **Explainability**: SHAP LinearExplainer for exact attributions
