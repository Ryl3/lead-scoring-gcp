# Project Decisions Log

Architectural and analytical decisions made during the project, with rationale.
Updated at the end of each stage.

---

## Stage 1 — EDA & Data Audit

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **EXCLUDE `crm_stage` from features** | 49 'Closed-Won' records conflict with outcomes — confirmed data leakage |
| 2 | **EXCLUDE bot suspects from training** | 40 records (5%) flagged as bots; contaminate behavioral signals |
| 3 | **EXCLUDE ghost trials from training** | 61 records (7.6%); ghost trial CVR indistinguishable from noise |
| 4 | **Proceed with Logistic Regression** | 195+ clean conversions exceed the 100-conversion threshold |
| 5 | **Time decay window = 30 days** | Median sales cycle = 13 days; 30d gives buffer without penalising valid long-cycle leads |
| 6 | **Clamp decay at 0, not negative** | Leads >30d inactive may be long-cycle, not cold; negative scores would incorrectly penalise them |
| 7 | **Impute CRM nulls with group medians** | Missing revenue & job_title show no significant CVR difference — Missing at Random |
| 8 | **Build trial engagement composite** | Raw trial start is weak (1.07x lift); engagement ratio (logins × features / days) captures quality |
| 9 | **Asymmetric treatment for negatives** | Unsubscribe (0.34x), careers page (0.79x) are disqualifiers, not just low-positive signals |

---

## Stage 2 — Feature Engineering
*(to be filled)*

## Stage 3 — Rule-Based Baseline
The point system is built from five signal families, weighted by their correlation strength from Stage 2:
- Behavioural (website) — the heaviest weights, because these signals are the hardest to fake and most predictive. Pricing page visits earn up to 24 points; demo page visits up to 18. The decayed versions are used throughout, so a visit from 6 weeks ago contributes less than one from yesterday.
- Email engagement — click behaviour gets more weight than opens (clicks require intent; opens can be accidental or auto-triggered). Click-to-open rate is also scored, since it reflects content relevance.
- Interaction terms — because Stage 2 showed these have outsized correlation (+0.280 for dm_x_pricing_or_demo), they're treated as bonus multipliers: a decision-maker who visited the pricing page earns +12 on top of the individual signals.
- Negative signals — careers page visits, unsubscribes, hard bounces, and bot suspects are penalised asymmetrically, not just scored zero. An unsubscribe costs –20 points.

#### Performance Metrics
Metric Value 
AUC-ROC 0.739
Average Precision 0.494
Top 20% lead capture 42% of all conversions

- An AUC of 0.739 from a hand-crafted rule system, before any ML, is a solid baseline. Random would give 0.50; a well-tuned model on this data should be in the 0.80–0.85 range — so there's clear headroom for Stage 4 to add lift, but the rule-based system is already meaningful.

#### Tier Breakdown
Tier    Leads   Conv. Rate  Conversions Captured
Hot     94 (12%)  58.5% 27%
Warm    210 (26%)   36.2%  37%
Nurture 313 (39%)   18.5%  28%
Suppress    183 (23%)   8.2%    7%


- The Hot tier converts at more than double the overall rate (25.5%). Hot + Warm together represent 38% of leads but capture 64% of conversions — meaning a sales team following this system could spend less than half their time and still reach nearly two-thirds of the deals. That's the business case in one number.

## Stage 4 — Predictive Modeling
Decisions & Rationale
- C=0.1 (stronger regularisation on LR) — With 33 features on 800 leads, default C=1.0 would allow the model to over-lean on any single feature. C=0.1 forces the model to spread weight across the feature set, which produces more stable coefficients and better generalization.
- class_weight='balanced' instead of SMOTE — At 25.5% conversion we're not in severe imbalance territory. SMOTE generates synthetic samples by interpolating between minority-class observations, which can introduce unrealistic data points on a dataset this small. Balanced class weights are a cleaner correction with no risk of artificial inflation.
- RF max_depth=6, min_samples_leaf=10 — Deliberately conservative constraints. Without them, a Random Forest on 800 leads would memorise training folds and produce misleading CV results. The constraint is what makes the RF comparison honest and comparable.
- LR wins over RF — LR achieves higher AUC (0.789 vs 0.777) with equal conversion capture. The reason is dataset size: RF's ability to learn non-linear interactions needs more data to outperform a regularised linear model. At 800 leads, the added complexity isn't justified. LR is the production model for Stage 5.
- 5-fold CV over train/test split — 160 holdout leads would give noisy, unstable metrics. CV uses all 800 leads for evaluation while preserving the no-leakage guarantee. All scores in stage4_model_scores.csv are honest out-of-fold predictions.
- Output includes an ensemble_prob column (0.6 LR / 0.4 RF blend) — Provided for Stage 5 experimentation, but LR is the primary signal. The blend is available if you want to test whether it meaningfully shifts any borderline tier assignments.

## Stage 5 — Output Layer

