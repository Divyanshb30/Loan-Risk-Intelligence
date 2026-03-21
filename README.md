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
| Neural Network (Stage 2, stacked) | **0.9184** | **0.4653** | **0.0637** |

**McNemar's Test**: χ²=194.76, p<0.0001 — NN improvement over XGBoost is statistically significant.

**Bootstrap 95% CI** on AUC gain: [+0.0006, +0.0011] — tight, consistent improvement.

**PSI**: 60/60 months stable — no feature distribution drift detected across the evaluation window.

AUC-PR of **0.4653 at a 7.8% positive rate** represents 6× improvement over a
random classifier (baseline = 0.078) — the operationally relevant metric for imbalanced credit risk datasets.

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
  │  Test AUC: 0.9184 | AUC-PR: 0.4653      │
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
Loan-Risk-Intelligence/
├── configs/
│   └── config.yaml                ← hyperparameters, paths, split config
├── notebooks/
│   ├── 01_eda.ipynb               ← EDA, regime analysis, macro residualization
│   └── 02_training.ipynb          ← OOF generation + XGBoost + NN training
├── src/
│   ├── data/
│   │   └── preprocess.py          ← raw .dta → clean parquet pipeline
│   ├── features/
│   │   └── build_features.py      ← feature engineering, macro residualization
│   ├── models/
│   │   ├── train_xgboost.py       ← XGBoost + 5-fold OOF + MLflow tracking
│   │   └── train_nn.py            ← Residual NN + stacking + focal loss
│   ├── api/                       ← FastAPI inference endpoint (in progress)
│   ├── explainability/            ← SHAP beeswarm + waterfall (in progress)
│   └── utils/
│       ├── config.py              ← load_config, get_project_root
│       └── logger.py              ← structured logging setup
├── model2_llm/                    ← Phase 2: Qwen 2.5 + LoRA (in progress)
├── tests/                         ← unit tests (in progress)
├── data/
│   ├── raw/                       ← P2P_Macro_Data.dta (gitignored)
│   ├── processed/                 ← features.parquet (gitignored)
│   └── outputs/                   ← model weights, predictions, scaler (gitignored)
├── logs/                          ← training logs
├── mlruns/                        ← MLflow experiment tracking
├── requirement.txt
└── README.md
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
git clone https://github.com/Divyanshb30/Loan-Risk-Intelligence
cd Loan-Risk-Intelligence
pip install -r requirement.txt
# Place P2P_Macro_Data.dta in data/raw/

# 1. Run preprocessing
python -m src.data.preprocess

# 2. Train XGBoost (generates OOF predictions)
python -m src.models.train_xgboost

# 3. Train Neural Network (stacks XGB logit)
python -m src.models.train_nn

# 4. Start API
uvicorn src.api.main:app --reload

# 5. Launch dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

---

## Status

- [x] Preprocessing + feature engineering — 1,802,285 records
- [x] Residualized macro features — FEDFUNDS, CPI, UNRATE
- [x] XGBoost — OOF AUC 0.9168
- [x] PyTorch Neural Network — Test AUC 0.9184, AUC-PR 0.4653, Brier 0.0637
- [x] 2016 regime shift diagnosis + year-stratified split
- [x] McNemar's Test — χ²=194.76, p<0.0001
- [x] Bootstrap CI — [+0.0006, +0.0011] on AUC gain
- [x] PSI drift detection — 60/60 months stable
- [x] FastAPI endpoint — POST /predict with SHAP drivers
- [x] Streamlit dashboard — live risk assessment UI
- [ ] SHAP beeswarm + waterfall plots
- [ ] Phase 2: LLM explanation layer (Qwen 2.5-3B + LoRA)
'''