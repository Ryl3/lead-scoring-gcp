"""FastAPI app configuration."""

from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT / "models" / "lead_scorer_v1.pkl"

# API config
APP_TITLE = "Lead Scoring API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = """
ML-powered lead scoring system with SHAP explainability.

## Endpoints

* **/health** - Health check
* **/score/batch** - Score batch of leads from CSV
* **/score/lead** - Score single lead from JSON
* **/model/info** - Model metadata
"""
