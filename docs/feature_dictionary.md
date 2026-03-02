# Feature Dictionary

Documents every feature in the model: source, transformation, and rationale.
Updated during Stage 2.

---

## Behavioral Features (time-decayed)

| Feature | Source | Formula | Rationale |
|---------|--------|---------|-----------|
| `pricing_page_decay` | website_analytics | `pricing_page_visits × max(0, 1 - days_since_last_visit/30)` | #1 signal (1.75x lift); decay neutralises stale activity |
| `demo_page_decay` | website_analytics | `demo_page_visits × max(0, 1 - days_since_last_visit/30)` | #2 signal (1.70x lift) |
| `behavioral_intent_score` | website_analytics + email | `pricing_page_decay + demo_page_decay + roi_calc×5 + forms×8` | Composite high-intent index |
| `email_engagement_ratio` | email_marketing | `emails_clicked / max(1, emails_sent)` | Normalises clicks across varying send volumes |
| `trial_engagement_score` | product_trial | `(trial_logins × features_activated) / max(1, days_active_in_trial)` if not ghost | Rewards depth over breadth |

## Firmographic Features

| Feature | Source | Encoding | Rationale |
|---------|--------|----------|-----------|
| `industry_tier` | crm_leads | Ordinal 1–3 from EDA CVR table | Finance/Logistics/Education=3; SaaS/Healthcare=2; others=1 |
| `company_size_score` | crm_leads | Ordinal 0–4 | Smaller companies convert better (1–10 emp = 4) |
| `lead_source_quality` | crm_leads | Ordinal 0–3 | Paid Social/Referral=3 → Cold Outreach=0 |
| `is_decision_maker` | crm_leads | Binary (as-is) | 1.11x lift; authority matters in SMB |
| `revenue_imputed` | crm_leads | Fill NaN with industry×size group median | Informed imputation avoids global median bias |

## Negative / Disqualifier Features

| Feature | Source | Formula | Rationale |
|---------|--------|---------|-----------|
| `negative_signal_flag` | email + website | `1 if unsubscribed=1 OR hard_bounced=1 OR careers_page_visits>2` | Binary disqualifier — asymmetric treatment |

## Excluded Features

| Feature | Reason |
|---------|--------|
| `crm_stage` | Data leakage — 49 label conflicts with outcomes |
| `is_bot_suspect` | Contamination flag — excluded from training entirely |
| `is_ghost_trial` | Contamination flag — excluded from training entirely |
| `lead_id` | Identifier — no predictive value |
| `created_date` | Not available at scoring time in production |
