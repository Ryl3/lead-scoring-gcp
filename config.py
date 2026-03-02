"""
Central configuration for the Lead Scoring project.
All paths, thresholds, and hyperparameters live here.
Notebooks and src modules import from this file — never hardcode values elsewhere.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
NOTEBOOKS   = ROOT / "notebooks"
SRC         = ROOT / "src"
OUT_REPORTS = ROOT / "outputs" / "reports"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_MODELS  = ROOT / "outputs" / "models"
OUT_SCORES  = ROOT / "outputs" / "scores"
DOCS        = ROOT / "docs"

# ── Raw file names ─────────────────────────────────────────────────────────
RAW_FILES = {
    "crm":      DATA_RAW / "crm_leads.csv",
    "web":      DATA_RAW / "website_analytics.csv",
    "email":    DATA_RAW / "email_marketing.csv",
    "trial":    DATA_RAW / "product_trial.csv",
    "outcomes": DATA_RAW / "outcomes_ground_truth.csv",
}

# ── Processed artefacts ────────────────────────────────────────────────────
FEATURE_MATRIX = DATA_PROC / "feature_matrix.csv"
TRAIN_SET      = DATA_PROC / "train.csv"
TEST_SET       = DATA_PROC / "test.csv"

# ── Data quality — exclusion flags ────────────────────────────────────────
# Leads matching these conditions are removed from training (kept for scoring)
EXCLUDE_FLAGS = {
    "is_bot_suspect": 1,
    "is_ghost_trial": 1,
}

# ── Feature engineering ────────────────────────────────────────────────────
DECAY_WINDOW_DAYS    = 30      # Linear decay window for behavioral scores
TRIAL_GHOST_EXCLUDE  = True    # Exclude ghost trials from engagement score
EMAIL_LAPSE_DAYS     = 60      # Imputation value for missing days_since_last_email_open
                               # Treated as "lapsed" — no recent engagement assumed
SESSION_DEPTH_CAP    = 20      # Upper clip for pages_per_session (handles outliers)

# ── Behavioural signal groups ──────────────────────────────────────────────
# Used in feature engineering (Stage 2) and rule-based scoring (Stage 3)
HIGH_INTENT_EVENTS   = ["pricing_page_visits", "demo_page_visits"]
MEDIUM_INTENT_EVENTS = ["case_study_views", "roi_calculator_uses"]
LOW_INTENT_EVENTS    = ["blog_views"]
NEGATIVE_EVENTS      = ["careers_page_visits"]

# ── Firmographic lookup tables ─────────────────────────────────────────────
INDUSTRY_TIERS = {
    3: ["Finance", "Logistics", "Education"],
    2: ["SaaS", "Healthcare"],
    1: [],   # all others → default tier 1
}

COMPANY_SIZE_SCORES = {
    "1-10":    4,
    "11-50":   3,
    "51-200":  2,
    "201-500": 1,
    "500+":    0,
}

LEAD_SOURCE_QUALITY = {
    3: ["Paid Social", "Referral"],
    2: ["Direct", "Content/Blog", "Organic Search"],
    1: ["Events", "Webinar"],
    0: ["Cold Outreach"],
}

# Job title → seniority tier (used in feature engineering and Stage 5 explainability)
SENIORITY_MAP = {
    "CEO": "executive", "Founder": "executive", "CTO": "executive",
    "VP Sales": "vp",   "VP Marketing": "vp",
    "Director of Operations": "director",
    "IT Manager": "manager", "Sales Manager": "manager",
    "Marketing Manager": "manager", "Unknown": "unknown",
}

# Tech stack → digital maturity score (proxy for budget and integration readiness)
TECH_STACK_SCORE = {
    "HubSpot|Slack|Google Workspace": 2,   # modern, digitally mature
    "Salesforce|Jira|AWS":            2,   # enterprise-leaning, tech-forward
    "Pipedrive|Notion|Shopify":       1,   # SMB-appropriate
    "Zoho|Trello|WooCommerce":        1,   # SMB, cost-conscious
    "None detected":                  0,
}

# ── Rule-based scoring weights (Stage 3) ──────────────────────────────────
RULE_WEIGHTS = {
    "pricing_page_visits":  20,
    "demo_page_visits":     20,
    "forms_submitted":      15,
    "roi_calculator_uses":  10,
    "case_study_views":     10,
    "content_downloads":     8,
    "webinar_attended":      5,
    "blog_views_3plus":      5,
    "trial_started":        15,
    # Negative signals
    "unsubscribed":        -20,
    "hard_bounced":        -15,
    "careers_page_excess":  -10,  # careers_page_visits > 2
}

# ── Model hyperparameters (Stage 4) ───────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5

LOGISTIC_PARAMS = {
    "C": 1.0,
    "max_iter": 500,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# ── Action layer thresholds (Stage 5) ─────────────────────────────────────
# Updated based on model score distribution and business priorities
SCORE_TIERS = {
    "Hot":      (60, 100),   # Immediate outreach — 2hr SLA
    "Warm":     (42,  59),   # Accelerated nurture — 24hr SLA
    "Nurture":  (25,  41),   # Long-term nurture — weekly batch
    "Suppress": (  0,  24),  # Minimal nurturing — monthly review
}
