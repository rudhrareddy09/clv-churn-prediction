# Customer Lifetime Value & Churn Prediction

End-to-end ML pipeline for predicting customer lifetime value and identifying churn risk using the CDNOW dataset, built with `lifetimes` (BG/NBD + Gamma-Gamma) and `pandas`.

## Why It Matters
Businesses need to know which customers are worth investing retention budget in. This pipeline translates raw transactional RFM data into actionable CLV predictions and customer segments, mirroring real-world MLOps workflows used in marketing analytics and growth teams.

## Pipeline Overview
1. **Data Loading** — CDNOW summary dataset (2,357 customers)
2. **Feature Engineering** — recency ratios, purchase rates, value flags
3. **BG/NBD Model** — predicts future purchase probability
4. **Gamma-Gamma Model** — predicts monetary value per transaction
5. **CLV Prediction** — 12-month customer lifetime value
6. **Segmentation** — Low / Medium / High / VIP tiers

## Key Findings
- Average predicted 12-month CLV: **$403.81**
- 211 VIP customers (avg CLV $1,267.77)
- 59.9% of customers never made a repeat purchase
- Top customer predicted CLV: $7,698

## Tech Stack
Python, pandas, scikit-learn, lifetimes, BG/NBD, Gamma-Gamma

## Setup
\`\`\`bash
pip install -r requirements.txt
python src/pipeline.py
\`\`\`

## Output
Results written to `outputs/clv_predictions.csv` with per-customer CLV predictions and segment labels.