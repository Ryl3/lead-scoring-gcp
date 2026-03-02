"""
FastAPI application for lead scoring.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import io

from app.core.config import APP_TITLE, APP_VERSION, APP_DESCRIPTION, MODEL_PATH
from app.core.scorer import LeadScorer

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize scorer (lazy load on first request if model not ready)
scorer: Optional[LeadScorer] = None


def get_scorer() -> LeadScorer:
    """Get or initialize scorer."""
    global scorer
    if scorer is None:
        if not MODEL_PATH.exists():
            raise HTTPException(500, "Model not found. Run scripts/train_model.py first.")
        scorer = LeadScorer(MODEL_PATH)
    return scorer


# Pydantic models
class SingleLeadRequest(BaseModel):
    lead_id: str
    features: Dict[str, float]


class SingleLeadResponse(BaseModel):
    lead_id: str
    lr_prob: float
    score_0_100: float
    percentile: float
    tier: str
    sla: str
    reason_1: str
    reason_2: str
    reason_3: str
    top_negative: str


class BatchScoreResponse(BaseModel):
    total_scored: int
    tier_distribution: Dict[str, int]
    results: List[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    version: str
    feature_count: int
    tiers: Dict
    tier_slas: Dict


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    model_loaded = MODEL_PATH.exists() and scorer is not None
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        version=APP_VERSION,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Get model metadata."""
    s = get_scorer()
    return ModelInfoResponse(
        version=APP_VERSION,
        feature_count=len(s.feature_cols),
        tiers=s.score_tiers,
        tier_slas=s.tier_sla,
    )


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(file: UploadFile = File(...)):
    """
    Score a batch of leads from CSV upload.
    
    CSV must contain:
    - lead_id column
    - All feature columns used during training
    
    Returns scored leads with tiers and explanations.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files accepted")
    
    try:
        content = file.file.read()
        s = get_scorer()
        results_df = s.score_from_csv(content)
        
        # Calculate tier distribution
        tier_counts = results_df["tier"].value_counts().to_dict()
        
        return BatchScoreResponse(
            total_scored=len(results_df),
            tier_distribution=tier_counts,
            results=results_df.to_dict('records'),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Scoring failed: {str(e)}")


@app.post("/score/lead", response_model=SingleLeadResponse)
def score_single(request: SingleLeadRequest):
    """
    Score a single lead from JSON features.
    
    Features must include all model features.
    Missing features will be filled with 0.
    """
    try:
        s = get_scorer()
        result = s.score_single(request.features, request.lead_id)
        return SingleLeadResponse(**result.__dict__)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Scoring failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
