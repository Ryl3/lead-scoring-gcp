"""
Train and serialize the lead scoring model pipeline.
This creates a reusable model artifact for the API.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"
OUT_SCORES = ROOT / "outputs" / "scores"
MODELS_DIR = ROOT / "models"

# Model config
RANDOM_STATE = 42
LR_C = 0.1
CV_FOLDS = 5

# Score tiers
SCORE_TIERS = {
    "Hot": (60, 100),
    "Warm": (42, 59),
    "Nurture": (25, 41),
    "Suppress": (0, 24),
}

TIER_SLA = {
    "Hot": "2-hour outreach",
    "Warm": "24-hour outreach",
    "Nurture": "Weekly nurture batch",
    "Suppress": "Monthly review only",
}


def train_and_save_model():
    """Train model on full dataset and save pipeline + artifacts."""
    print("Loading data...")
    features = pd.read_csv(DATA_PROC / "feature_matrix.csv")
    scores = pd.read_csv(OUT_SCORES / "stage4_model_scores.csv")
    
    feature_cols = [c for c in features.columns if c != "lead_id"]
    
    # Merge with outcomes
    df = features.merge(scores[["lead_id", "converted"]], on="lead_id")
    X = df[feature_cols].fillna(0).values
    y = df["converted"].values
    
    print(f"Training on {len(X)} leads with {len(feature_cols)} features...")
    
    # Build and train pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=LR_C,
            max_iter=500,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
    
    pipeline.fit(X, y)
    
    # Validate
    train_probs = pipeline.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, train_probs)
    print(f"In-sample AUC: {auc:.3f}")
    
    # Build SHAP explainer
    scaler = pipeline.named_steps["scaler"]
    lr_model = pipeline.named_steps["lr"]
    X_scaled = scaler.transform(X)
    background = shap.maskers.Independent(X_scaled, max_samples=200)
    explainer = shap.LinearExplainer(lr_model, background)
    
    # Save artifacts
    MODELS_DIR.mkdir(exist_ok=True)
    
    artifacts = {
        "pipeline": pipeline,
        "explainer": explainer,
        "feature_cols": feature_cols,
        "score_tiers": SCORE_TIERS,
        "tier_sla": TIER_SLA,
        "random_state": RANDOM_STATE,
    }
    
    model_path = MODELS_DIR / "lead_scorer_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"  Size: {model_path.stat().st_size / 1024:.1f} KB")
    
    return artifacts


if __name__ == "__main__":
    train_and_save_model()
