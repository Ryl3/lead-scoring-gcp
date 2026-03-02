"""
Core scoring module for lead scoring.
Handles model loading, batch scoring, and SHAP explainability.
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import io

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# Feature labels for explainability
FEATURE_LABELS = {
    "high_intent_touch_count": "Multiple high-intent page visits",
    "visited_pricing_or_demo": "Visited pricing or demo page",
    "cross_channel_engaged": "Engaged across multiple channels",
    "dm_x_pricing_or_demo": "Decision-maker visited pricing/demo",
    "pricing_page_visits_decayed": "Recent pricing page visit",
    "content_engagement_decayed": "Recent content engagement",
    "smb_fit_x_intent": "Strong SMB fit + high intent",
    "good_source_x_content": "Quality lead source with content engagement",
    "demo_page_visits_decayed": "Recent demo page visit",
    "open_rate": "High email open rate",
    "email_clicks_decayed": "Recent email click activity",
    "email_opens_decayed": "Recent email opens",
    "industry_tier": "High-value industry",
    "lead_source_tier": "High-quality lead source",
    "click_to_open_rate": "High email click-to-open rate",
    "is_decision_maker": "Is a decision-maker",
    "case_study_views_decayed": "Recently viewed case studies",
    "company_size_score": "Ideal company size",
    "trial_depth_score": "Deep product trial engagement",
    "is_engaged_trial": "Active trial user",
    "is_smb_fit": "Fits SMB profile",
    "senior_x_trial": "Senior contact with trial activity",
    "seniority_rank": "Senior job title",
    "visited_last_7_days": "Visited site in last 7 days",
    "log_annual_revenue": "High annual revenue",
    "tech_stack_score": "Strong tech stack fit",
    "pages_per_session": "High page depth per session",
    "roi_calculator_uses_decayed": "Used ROI calculator recently",
    "blog_views_decayed": "High blog consumption",
    "is_unsubscribed": "Unsubscribed from emails ↓",
    "careers_page_visits_decayed": "Visited careers page ↓",
    "is_hard_bounced": "Email hard-bounced ↓",
    "is_bot_suspect": "Suspected bot activity ↓",
}


@dataclass
class ScoringResult:
    """Result container for scored leads."""
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


class LeadScorer:
    """Lead scoring model wrapper with SHAP explainability."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize scorer with saved model artifacts."""
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "lead_scorer_v1.pkl"
        
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)
        
        self.pipeline = artifacts["pipeline"]
        self.explainer = artifacts["explainer"]
        self.feature_cols = artifacts["feature_cols"]
        self.score_tiers = artifacts["score_tiers"]
        self.tier_sla = artifacts["tier_sla"]
        
        self.scaler = self.pipeline.named_steps["scaler"]
        self.lr_model = self.pipeline.named_steps["lr"]
    
    def _assign_tier(self, score: float) -> str:
        """Assign tier based on 0-100 score."""
        for tier, (lo, hi) in self.score_tiers.items():
            if lo <= score <= hi:
                return tier
        return "Suppress"
    
    def _get_explanations(self, shap_values: np.ndarray) -> Tuple[str, str, str, str]:
        """Extract top 3 positive and top negative reasons from SHAP values."""
        sv = shap_values
        
        # Positive contributors
        pos_idx = np.argsort(sv)[::-1]
        pos_reasons = []
        for i in pos_idx:
            if sv[i] > 0.01:
                feat_name = self.feature_cols[i]
                pos_reasons.append(FEATURE_LABELS.get(feat_name, feat_name))
            if len(pos_reasons) == 3:
                break
        
        # Pad to 3 if needed
        while len(pos_reasons) < 3:
            pos_reasons.append("—")
        
        # Top negative contributor
        neg_idx = np.argsort(sv)
        top_neg = "—"
        if sv[neg_idx[0]] < -0.01:
            feat_name = self.feature_cols[neg_idx[0]]
            top_neg = FEATURE_LABELS.get(feat_name, feat_name)
        
        return pos_reasons[0], pos_reasons[1], pos_reasons[2], top_neg
    
    def score_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a batch of leads.
        
        Args:
            features_df: DataFrame with lead_id and feature columns
        
        Returns:
            DataFrame with scores, tiers, and explanations
        """
        # Ensure all required columns present
        missing_cols = set(self.feature_cols) - set(features_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract lead IDs
        if "lead_id" not in features_df.columns:
            raise ValueError("features_df must contain 'lead_id' column")
        
        lead_ids = features_df["lead_id"].values
        X = features_df[self.feature_cols].fillna(0).values
        
        # Get probabilities
        lr_probs = self.pipeline.predict_proba(X)[:, 1]
        
        # Convert to 0-100 score via percentile rescaling
        percentile_rank = rankdata(lr_probs, method="average") / len(lr_probs)
        score_0_100 = (percentile_rank * 100).round(1)
        
        # Get SHAP values
        X_scaled = self.scaler.transform(X)
        shap_values = self.explainer(X_scaled)
        
        # Generate explanations for each lead
        results = []
        for i in range(len(X)):
            r1, r2, r3, neg = self._get_explanations(shap_values.values[i])
            tier = self._assign_tier(score_0_100[i])
            
            results.append({
                "lead_id": lead_ids[i],
                "lr_prob": round(lr_probs[i], 4),
                "score_0_100": score_0_100[i],
                "percentile": round(percentile_rank[i] * 100, 1),
                "tier": tier,
                "sla": self.tier_sla[tier],
                "reason_1": r1,
                "reason_2": r2,
                "reason_3": r3,
                "top_negative": neg,
            })
        
        return pd.DataFrame(results)
    
    def score_single(self, features: Dict[str, float], lead_id: str = "unknown") -> ScoringResult:
        """
        Score a single lead from feature dictionary.
        
        Args:
            features: Dict of feature names to values
            lead_id: Lead identifier
        
        Returns:
            ScoringResult with score and explanations
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        df["lead_id"] = lead_id
        
        result_df = self.score_batch(df)
        row = result_df.iloc[0]
        
        return ScoringResult(
            lead_id=row["lead_id"],
            lr_prob=row["lr_prob"],
            score_0_100=row["score_0_100"],
            percentile=row["percentile"],
            tier=row["tier"],
            sla=row["sla"],
            reason_1=row["reason_1"],
            reason_2=row["reason_2"],
            reason_3=row["reason_3"],
            top_negative=row["top_negative"],
        )
    
    def score_from_csv(self, csv_content: bytes) -> pd.DataFrame:
        """
        Score leads from uploaded CSV content.
        
        Args:
            csv_content: Raw bytes of CSV file
        
        Returns:
            DataFrame with scores
        """
        df = pd.read_csv(io.BytesIO(csv_content))
        return self.score_batch(df)
