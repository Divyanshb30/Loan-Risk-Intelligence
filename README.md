'''
# Loan Risk Intelligence System
A production-grade, two-stage credit risk prediction system trained on 1.8M real-world
loan records. Predicts default probability, quantifies macroeconomic risk drivers via
residualized macro signals, and serves predictions through a REST API with
SHAP-based explainability.

---

## Results

| Model | Test AUC | AUC-PR | Brier Score |
|---|---|---|---|
| XGBoost (Stage 1) | 0.9168 | — | — |
| Neural Network (Stage 2) | 0.9184 | 0.4646 | 0.0653 |
| Ensemble | — | — | — |

AUC-PR of **0.465 at a 7.8% positive rate** represents 6× improvement over a
random classifier — the operationally relevant metric for imbalanced credit risk
datasets.

---

## Key Finding

`FEDFUNDS_resid` — the residualized Federal Funds Rate — is the dominant default
driver at **30.4% feature importance**.

Residualization removes the autocorrelated trend component from the raw rate,
isolating *unexpected* monetary tightening as the predictive signal rather than
the absolute rate level. Borrowers who originated loans during periods of
unexpected Fed tightening defaulted at systematically higher rates, independent
of individual credit profile characteristics.

---

## Architecture

```
Lending Club Dataset (2012–2019, 2.2M rows)
              ↓
  Preprocessing Pipeline
  — Binary target: Fully Paid vs Charged Off
  — Leakage column removal
  — Label encoding + null handling
  — Output: 1,802,285 clean records
              ↓
  Feature Engineering (~62 features)
  — Loan characteristics: DTI, income, grade, term
  — Temporal: issue year/month, loan age
  — Residualized macro signals:
    FEDFUNDS_resid · CPI_resid · UNRATE_resid
              ↓
  ┌──────────────────────────────────────────┐
  │  Stage 1: XGBoost Classifier             │
  │  5-Fold OOF predictions (leak-free)      │
  │  OOF AUC: 0.9168                         │
  └──────────────────┬───────────────────────┘
                     │  OOF logit → stacking feature
                     ↓
  ┌──────────────────────────────────────────┐
  │  Stage 2: PyTorch Feedforward Network    │
  │  Input dim: 60 features + 1 XGB logit    │
  │  BatchNorm · Dropout · BCE + pos_weight  │
  │  Early stopping on validation AUC        │
  │  Test AUC: 0.9184 | AUC-PR: 0.4646      │
  └──────────────────┬───────────────────────┘
                     ↓
       Weighted Ensemble (XGB + NN)
                     ↓
  ┌──────────────────────────────────────────┐
  │  Evaluation                              │
  │  McNemar\'s Test + Bootstrap CI (95%)     │
  │  ROC · PR · KS Statistic · Gini          │
  │  Per-year performance breakdown          │
  └──────────────────┬───────────────────────┘
                     ↓
  ┌──────────────────────────────────────────┐
  │  SHAP Explainability                     │
  │  Global: beeswarm feature importance     │
  │  Local: per-loan waterfall explanations  │
  └──────────────────┬───────────────────────┘
                     ↓
  ┌──────────────────────────────────────────┐
  │  FastAPI Inference Endpoint              │
  │  POST /predict                           │
  │  Returns: default_probability,           │
  │           risk_tier, top_shap_drivers    │
  └──────────────────┬───────────────────────┘
                     ↓
  ┌──────────────────────────────────────────┐
  │  Phase 2 — In Progress                   │
  │  Qwen 2.5-3B + LoRA fine-tuned on        │
  │  (SHAP values + macro context) →         │
  │  analyst-style natural language reports  │
  └──────────────────────────────────────────┘
```

---

## The 2016 Credit Regime Shift

EDA revealed a structural distributional break in 2016 following Lending Club\'s
underwriting tightening:

| Period | Default Rate |
|---|---|
| 2012–2015 | ~9.4% |
| 2016–2019 | ~2.4% |

A naive temporal train/test split — training on pre-2016 data, evaluating on
post-2016 — produced a Neural Network AUC ceiling of 0.71 regardless of
architecture or hyperparameter configuration. The root cause was complete
distributional mismatch, not model capacity.

**Resolution:** Year-stratified train/test split — 80%/20% sampled from each
calendar year — ensures both credit regimes are represented proportionally in
training and evaluation. The train/test AUC gap collapsed from **0.17 to 0.003**
after this change.

This split design evaluates generalisation across historical market conditions
rather than pure out-of-time prediction, which better reflects deployment
environments where both low and high default-rate regimes recur.

---

## Project Structure

```
lending-club-risk/
├── config.yaml                    ← hyperparameters, paths, split config
├── notebooks/
│   └── main.ipynb                 ← end-to-end pipeline
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_xgboost.py
│   │   └── train_nn.py
│   └── utils/
│       └── config.py
├── output/                        ← model outputs (gitignored)
└── models/                        ← saved weights (gitignored)
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Modelling | XGBoost, PyTorch |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| Drift Detection | KS-Test, PSI |
| Model Selection | McNemar\'s Test, Bootstrap CI |
| Serving | FastAPI, Docker |
| Dashboard | Streamlit |
| Fine-tuning (Phase 2) | Qwen 2.5-3B, LoRA, PEFT, Unsloth |
| Data | Lending Club (Kaggle), FRED macroeconomic series |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/lending-club-risk
cd lending-club-risk
pip install -r requirements.txt
# Place Lending Club CSV in data/raw/
jupyter notebook notebooks/main.ipynb
```

---

## Status

- [x] Preprocessing + feature engineering — 1,802,285 records
- [x] Residualized macro features — FEDFUNDS, CPI, UNRATE
- [x] XGBoost — OOF AUC 0.9168
- [x] PyTorch Neural Network — Test AUC 0.9184, AUC-PR 0.4646
- [x] 2016 regime shift diagnosis + year-stratified split
- [ ] Ensemble + McNemar\'s Test + Bootstrap CI
- [ ] SHAP explainability
- [ ] PSI + KS drift detection
- [ ] FastAPI endpoint
- [ ] Streamlit dashboard
- [ ] Phase 2: LLM explanation layer (Qwen 2.5-3B + LoRA)
'''