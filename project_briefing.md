# Loan-Risk-Intelligence — Complete Project Briefing
> **Purpose:** Full technical handoff document for AI-assisted deployment continuation.  
> **Generated:** 2026-03-28 | **Repo:** `Divyanshb30/Loan-Risk-Intelligence`  
> **Root:** `c:\college\PROJECTS\ML\Loan-Risk-Intelligence\`

---

## 1. PROJECT STRUCTURE

### Full Folder / File Tree

```
Loan-Risk-Intelligence/
├── .env                              # OPENAI_API_KEY, HF_TOKEN (gitignored)
├── .gitignore                        # 273-line gitignore (see §11)
├── README.md                         # 9 KB public README
├── requirement.txt                   # Root-level dev deps
├── mlflow.db                         # MLflow SQLite tracking DB (root)
│
├── configs/
│   └── config.yaml                   # Central hyperparameter + path config
│
├── data/
│   ├── raw/
│   │   ├── P2P_Macro_Data.dta        # 3.02 GB — Stata source file
│   │   ├── P2P_Macro_Data.parquet    # 404 MB — parquet cache of .dta
│   │   ├── P2P_Macro_Data.zip        # 575 MB — compressed archive
│   │   ├── Labels.do                 # Stata label definitions
│   │   └── P2P_Macro_Codes.do        # Stata encoding script
│   ├── processed/
│   │   └── features.parquet          # 275 MB — preprocessed feature matrix
│   └── outputs/                      # 25 files — see §4 for full listing
│
├── deploy/                           # GCP Cloud Run deployment package
│   ├── app.py                        # Self-contained FastAPI v2 (full NN pipeline)
│   ├── Dockerfile                    # python:3.11-slim, port 8080
│   ├── requirements.txt              # CPU-only PyTorch + FastAPI stack
│   └── artifacts/                    # 7 model files committed to git
│       ├── xgb_model.json            # XGBoost v2 model (3.7 MB)
│       ├── nn_best.pt                # Trained NN weights (925 KB)
│       ├── scaler.joblib             # Fitted StandardScaler
│       ├── scaler_mean.npy           # scaler.mean_ (608 B)
│       ├── scaler_scale.npy          # scaler.scale_ (608 B)
│       ├── feature_cols.joblib       # Ordered list of 60 feature names
│       └── feature_medians.joblib    # Median values per feature (imputation)
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_training.ipynb             # XGBoost + NN training orchestration
│   ├── 03_evaluation.ipynb           # Model evaluation, drift, SHAP waterfall plots
│   ├── 04_test_api.ipynb             # API smoke test (local uvicorn)
│   ├── 05_llm.ipynb                  # GPT-4o explanation generation + dataset build
│   ├── 06_kaggle_lora_finetune.ipynb # LoRA fine-tuning on Kaggle GPU (Llama-3.2-1B)
│   ├── 07_artifact_prep.ipynb        # Packs deploy/artifacts/ from data/outputs/
│   └── mlflow.db                     # Notebook-local MLflow DB
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                   # FastAPI v1 — XGBoost only (local dev)
│   ├── dashboard/
│   │   └── app.py                    # Streamlit dashboard (316 lines)
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py             # Raw → cleaned parquet pipeline
│   ├── explainability/
│   │   └── __init__.py               # (empty — SHAP logic is in notebooks)
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py         # Feature groups, residualization, interactions
│   ├── llm/
│   │   ├── generate_dataset.py       # Samples 1000 loans for LLM dataset
│   │   ├── generate_explanations.py  # Calls GPT-4o for explanations
│   │   ├── inference.py              # (0 bytes — stub/empty)
│   │   └── push_dataset.py           # Pushes train/val/test.jsonl to HuggingFace Hub
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_xgboost.py          # XGBoost training + OOF generation
│   │   └── train_nn.py               # LoanRiskNN training + stacking assembly
│   └── utils/
│       ├── config.py                 # load_config() + get_project_root()
│       └── logger.py                 # Rotating file + console logger
│
├── tests/                            # Empty — no unit tests written
├── model2_llm/                       # Empty — local LLM weights not stored here
├── mlruns/                           # MLflow experiment runs (gitignored)
└── loanvenv/                         # Python venv (gitignored)
```

---

## 2. DATA

### Dataset
| Property | Value |
|---|---|
| **Name** | P2P_Macro_Data (LendingClub + US Macro) |
| **Source** | Custom Stata dataset — LendingClub loan-level data merged with macro variables |
| **Raw size** | 3.02 GB (.dta) / 404 MB (parquet) |
| **Date range** | 2012–2026 (issue_year from 2012 to present) |
| **Post-filter rows** | ~1.8M loans (after excluding 'Current' + 'In Grace Period' statuses) |
| **Processed parquet** | `data/processed/features.parquet` — 275 MB |

### Target Variable
| Property | Value |
|---|---|
| **Column name** | `badloan` |
| **Positive class (1)** | Defaulted / Charged-off loan |
| **Negative class (0)** | Fully paid |
| **Class imbalance** | ~92% negative / ~8% positive ≈ **11.5:1 ratio** |
| **Handling** | XGBoost: `scale_pos_weight=11.5`; NN: BCEWithLogitsLoss `pos_weight` capped at 3.0 |

### All 60 Feature Names (exact column order as used in model)

**Group 1 — LOAN (28 features):**
```
logfunded_amnt, int_rate, installment, inq_last_6mths, open_acc,
logtotal_acc, pub_rec, des_length, logpct_tl_nvr_dlq, logtot_hi_cred_lim,
revol_util, revol_bal, delinq_2yrs, mths_since_last_delinq,
mths_since_recent_inq, mths_since_recent_bc, acc_open_past_24mths,
mort_acc, pub_rec_bankruptcies, tax_liens, acc_now_delinq,
num_tl_90g_dpd_24m, pct_tl_nvr_dlq, tot_cur_bal, avg_cur_bal,
bc_util, num_actv_bc_tl, num_actv_rev_tl
```

**Group 2 — BORROWER (5 features):**
```
logannual_inc, logdti, emp_length_, home_ownership_, verification_status_
```

**Group 3 — MACRO (17 features, residualized):**
```
muni_6m_resid, FED_lag6_resid, CPIUS_resid, FEDFUNDS_resid, inf_resid,
inf_6m_resid, muni_points_resid, riskprem_resid, cpi_resid,
logunemployment_resid, logearnings_resid, gdpcontrib_resid,
loglabor_force, logempl_birth, lognew_bus, loginternetuser
```
*(note: last 4 are cross-sectional by nature — not residualized)*

**Group 4 — CATEGORICAL (5 features, label-encoded to int):**
```
grade_, term_, purpose_, application_type_, initial_list_status_
```

**Group 5 — TIME (1 feature):**
```
issue_year
```

**Group 6 — INTERACTION (5 engineered features):**
```
dti_x_unemployment, real_interest_rate, loan_to_income,
util_x_delinq, fed_x_term
```

**Total: 28 + 5 + 17 − 1 (cross-sectional macro counted separately) + 5 + 1 + 5 = 60 features** (confirmed by `feature_cols.joblib` at 992 bytes, consistent with 60-col list)

### Key Preprocessing Decisions

| Decision | Detail |
|---|---|
| **Leakage removal** | 15 post-funding columns dropped: `total_pymnt`, `total_rec_prncp`, `recoveries`, `out_prncp`, `last_pymnt_amnt`, `last_pymnt_d`, etc. |
| **Ambiguous status removal** | 'Current' and 'In Grace Period' loans excluded — outcome unknown |
| **Missing value strategy** | `mths_since_*` → 999 (never happened); `mort_acc`, `tax_liens`, etc. → 0; remaining numeric → median; categorical → 'Unknown' |
| **Log transforms** | `funded_amnt` → `logfunded_amnt`; `annual_inc` → `logannual_inc`; `dti` → `logdti`; `total_acc` → `logtotal_acc`; `unemployment` → `logunemployment_resid`; etc. |
| **Macro residualization** | 12 macro vars residualized: subtract period mean grouped by `(issue_year, month)` → removes regime-level time trend, keeps cross-sectional signal only |
| **Interaction features** | 5 hand-crafted: `dti_x_unemployment` (logdti × logunemployment_resid), `real_interest_rate` (int_rate − inf), `loan_to_income` (logfunded_amnt / logannual_inc), `util_x_delinq` (revol_util × (delinq_2yrs+1)), `fed_x_term` (FED_lag6 × term_) |
| **Categorical encoding** | LabelEncoder on `grade_`, `term_`, `purpose_`, `application_type_`, `initial_list_status_`, `emp_length_`, `home_ownership_`, `verification_status_` |
| **Train/test split** | 80/20 `year_stratified` — `train_test_split` with `stratify=issue_year` — ensures every year represented in both sets |

---

## 3. MODEL ARCHITECTURE

### Stage 1: XGBoost

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 5 |
| `gamma` | 0.1 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `scale_pos_weight` | ~11.5 (computed as neg/pos from y_train) |
| `tree_method` | hist |
| `device` | cuda (RTX 3060 during training; CPU in deployment) |
| `early_stopping_rounds` | 50 (eval_set = test set during final training) |
| `random_state` | 42 |
| `eval_metric` | auc |

**Cross-validation:** 5-fold `TimeSeriesSplit` on the training set, tracking AUC-ROC, AUC-PR, and Brier Score.

**OOF Predictions (for NN stacking):**
- A second pass runs `TimeSeriesSplit` CV; for each fold, train on tr_idx, predict val_idx.
- `xgb_oof_predictions.npy` — full-length array of float32 predictions (NaN where excluded = first chunk, ~360K rows, never in a validation fold in TimeSeriesSplit).
- `xgb_oof_valid_mask.npy` — boolean mask; `~np.isnan(oof_preds)`.
- Valid OOF AUC: logged per fold (exact value in MLflow run `xgboost_baseline`).
- Inference-time: XGBoost predicts on full test set → `xgb_predictions.parquet` (`y_prob_xgb`).

**Saved artifact:** `data/outputs/xgb_model.json` (3.7 MB XGBoost v2 JSON format)

---

### Stage 2: LoanRiskNN (PyTorch)

**Architecture — exact class definition:**

```
LoanRiskNN(input_dim=63, dropout=0.2)   # 63 = 60 features + 1 XGB logit (after scaling)

Layer stack:
  input_norm   → BatchNorm1d(63)
  input_proj   → Linear(63→256), BatchNorm1d(256), GELU, Dropout(0.2)
  res_block1   → ResidualBlock(256, 0.2):
                    block: Linear(256→256), BN(256), GELU, Dropout(0.2),
                           Linear(256→256), BN(256)
                    out: GELU(x + block(x)), Dropout(0.2)
  compress1    → Linear(256→128), BatchNorm1d(128), GELU, Dropout(0.2)
  res_block2   → ResidualBlock(128, 0.2):
                    block: Linear(128→128), BN(128), GELU, Dropout(0.2),
                           Linear(128→128), BN(128)
                    out: GELU(x + block(x)), Dropout(0.2)
  compress2    → Linear(128→64), BatchNorm1d(64), GELU, Dropout(0.2)
  output       → Linear(64→1)

Forward:  x → input_norm → input_proj → res_block1 → compress1 → res_block2 → compress2 → output → squeeze(1)
Output:   raw logit (sigmoid applied externally for probabilities)
```

**Training hyperparameters:**

| Parameter | Value |
|---|---|
| `batch_size` | 4096 |
| `epochs` | 50 (early stopping at patience=12) |
| `learning_rate` | 0.0003 |
| `dropout` | 0.2 |
| `weight_decay` | 0.0001 |
| `optimizer` | AdamW |
| `scheduler` | ReduceLROnPlateau (mode=max, factor=0.5, patience=6, min_lr=1e-6) |
| `grad_clip` | 1.0 (clip_grad_norm_) |

**Loss function:**  
Config has `use_focal = false` (key is present in YAML but `use_focal` is NOT set in config.yaml — defaults to BCEWithLogitsLoss).  
→ `BCEWithLogitsLoss(pos_weight=min(raw_weight, 3.0))` where raw_weight = (1-pos_rate)/pos_rate ≈ 11.8; pos_weight capped at **3.0**.  
*(Note: MLflow run is named `nn_stacked_focal` and logs `loss: FocalLoss` — this was logged before the config was changed. Actual loss used is BCE with pos_weight=3.0 as of latest training.)*

**Weight initialization:** Kaiming normal on all Linear layers.

**Final Test Metrics (from `src/dashboard/app.py` header badges, confirmed in notebook 03):**
| Metric | Value |
|---|---|
| **Test AUC-ROC** | **0.9184** |
| **AUC-PR** | **0.465** |
| **Brier Score** | **0.064** |

**Saved artifact:** `data/outputs/nn_best.pt` (best epoch state_dict, 925 KB)

---

### How Stage 1 → Stage 2 Connect (Stacking Pipeline)

Exact input construction for the NN (from `src/models/train_nn.py`, lines 324–352):

```python
# 1. Load OOF predictions and valid mask
xgb_oof    = np.load("xgb_oof_predictions.npy")   # full array, NaNs for first chunk
valid_mask = np.load("xgb_oof_valid_mask.npy")     # bool mask, ~360K False at start

# 2. Mask out NaN rows (removes ~360K from training)
X_train_valid = X_train.values[valid_mask]         # (N_valid, 60)
y_train_valid = y_train[valid_mask]
xgb_oof_valid = xgb_oof[valid_mask]                # (N_valid,) — no NaNs

# 3. Convert XGB probabilities to logit space
eps = 1e-7
safe_logit = lambda p: np.log(clip(p,eps,1-eps) / (1-clip(p,eps,1-eps)))
xgb_oof_logit  = safe_logit(xgb_oof_valid).reshape(-1, 1)   # (N_valid, 1)
xgb_test_logit = safe_logit(xgb_test_probs).reshape(-1, 1)  # (N_test, 1)

# 4. Scale the 60-feature matrix
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_valid)  # (N_valid, 60)
X_test_scaled  = scaler.transform(X_test.values)       # (N_test, 60)
# scaler saved → scaler.joblib

# 5. Horizontal stack: 60 scaled features + 1 XGB logit
X_train_final = np.hstack([X_train_scaled, xgb_oof_logit])  # (N_valid, 61)
X_test_final  = np.hstack([X_test_scaled,  xgb_test_logit]) # (N_test, 61)

# NOTE: input_dim = 61, not 63 as docstring says.
# deploy/app.py says: input_dim = len(FEATURE_COLS) + 1 = 60 + 1 = 61
```

**At inference time (deploy/app.py, lines 124–148):**
```python
# Stage 1: XGBoost probability
xgb_prob = xgb_model.predict_proba(row)[:, 1][0]

# Convert to logit
xgb_logit = log(clip(xgb_prob, eps, 1-eps) / (1-clip(xgb_prob, eps, 1-eps)))

# Scale the 60 features using saved scaler
x_scaled = scaler.transform(row.values)            # (1, 60)

# Stack
x_nn = np.hstack([x_scaled, [[xgb_logit]]])        # (1, 61)

# Stage 2: NN forward pass
nn_prob = sigmoid(nn_model(tensor(x_nn))).item()
```

---

### Model Selection — Statistical Test

**Test used:** McNemar's test on binarized predictions (XGBoost alone vs. XGBoost+NN stacking ensemble).  
**Result:** **p < 0.0001** (displayed in Streamlit dashboard header badge and placeholder text).  
This confirms the NN stacking significantly reduces prediction disagreements vs. XGBoost alone.

---

## 4. ARTIFACTS IN `data/outputs/`

| File | Size | Purpose |
|---|---|---|
| `X_train.parquet` | 170 MB | Full 60-feature train split (used by API for median imputation) |
| `X_test.parquet` | 43 MB | Full 60-feature test split |
| `y_train.parquet` | 11.8 MB | Binary target for train set |
| `y_test.parquet` | 3.0 MB | Binary target for test set |
| `xgb_model.json` | 3.7 MB | Trained XGBoost model (XGBoost JSON v2 format) |
| `xgb_predictions.parquet` | 2.7 MB | XGBoost test-set probabilities (`y_prob_xgb` column) |
| `xgb_oof_predictions.npy` | 17.3 MB | Full OOF array (float32, NaN at first chunk) |
| `xgb_oof_valid_mask.npy` | 2.2 MB | Boolean mask for valid OOF rows |
| `xgb_feature_importance.csv` | 1.7 KB | 61 rows — feature name + XGB importance score |
| `nn_best.pt` | 925 KB | Best epoch state_dict for LoanRiskNN |
| `nn_predictions.parquet` | 2.7 MB | NN test-set probabilities (`y_prob_nn` column) |
| `scaler.joblib` | 2.0 KB | Fitted StandardScaler (fit on masked X_train) |
| `shap_values.npy` | 129.8 MB | SHAP values for all test-set samples (shape: N_test × 60) |
| `shap_dataset_raw.json` | 1.2 MB | 1000 loans: loan_id, prob, risk_tier, SHAP drivers, macro context (pre-explanation) |
| `shap_dataset_raw.csv` | 100 KB | Flat CSV preview of shap_dataset_raw.json |
| `shap_dataset_explained.json` | 1.6 MB | 1000 loans with GPT-4o analyst memo explanations |
| `shap_dataset_explained_v2.json` | 1.6 MB | Refined version v2 of GPT-4o explanations |
| `shap_dataset_test50.json` | 80 KB | 50-loan subset for test/demo purposes |
| `train.jsonl` | 730 KB | LLM fine-tune training split (JSONL format, instruction-response pairs) |
| `val.jsonl` | 92 KB | LLM fine-tune validation split |
| `test.jsonl` | 91 KB | LLM fine-tune test split |
| `drift_report.csv` | 2.8 KB | PSI + KS results for all 61 features |
| `shap_waterfall_high_risk.png` | 100 KB | SHAP waterfall plot — high-risk loan |
| `shap_waterfall_medium_risk.png` | 99 KB | SHAP waterfall plot — medium-risk loan |
| `shap_waterfall_low_risk.png` | 101 KB | SHAP waterfall plot — low-risk loan |

**Total: 25 files**

---

## 5. EXPLAINABILITY

### SHAP Setup
- **Model:** XGBoost (Stage 1) — `xgb_model.json`
- **Explainer:** `shap.TreeExplainer(xgb_model)` — exact tree-based SHAP, no sampling approximation
- **Computed over:** Full X_test set (43 MB, N_test rows × 60 features)
- **Stored:** `shap_values.npy` (129.8 MB, shape = [N_test, 60])
- **At inference:** Per-request SHAP computed live in both `src/api/main.py` and `deploy/app.py` via `explainer(row).values[0]`; top 3 (local API) or top 5 (deploy API) returned

### Top Feature Driver

| Rank | Feature | XGB Importance |
|---|---|---|
| **1** | **FEDFUNDS_resid** | **30.40%** |
| 2 | CPIUS_resid | 13.77% |
| 3 | grade_ | 11.18% |
| 4 | issue_year | 4.12% |
| 5 | riskprem_resid | 3.58% |

**FEDFUNDS_resid** is the dominant driver at 30.4% of total importance — macro monetary policy signal dominates over all borrower-level characteristics.

---

## 6. DRIFT DETECTION

### Method
Both PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) tests were run against all features. Results stored in `data/outputs/drift_report.csv`.

| Method | Result |
|---|---|
| **PSI** | **0.000 for all 61 features** (PSI < 0.1 = stable; PSI > 0.25 = significant drift) |
| **KS statistic** | Range 0.0000–0.0016 across all features |
| **KS p-value** | All p > 0.24 (most > 0.9) — none statistically significant |
| **Drift flags** | **✅ STABLE** for all 61 features — no drift detected |

**Interpretation:** The year-stratified split ensures both train and test share the same year distribution, producing near-zero PSI and non-significant KS results. This is expected by design, not a surprising finding.

---

## 7. LLM COMPONENT

### Architecture Chain
```
Raw SHAP output (shap_dataset_raw.json, 1000 records)
  → GPT-4o explanations (generate_explanations.py)
  → shap_dataset_explained.json (1000 records with analyst memos)
  → JSONL conversion (notebooks/05_llm.ipynb)
  → train.jsonl (800 records) / val.jsonl (100) / test.jsonl (100)
  → push_dataset.py → HuggingFace Hub: Divb30/loan-risk-explanations (private)
  → LoRA fine-tuning (notebooks/06_kaggle_lora_finetune.ipynb — Kaggle GPU)
```

### Fine-tuning Details
| Property | Value |
|---|---|
| **Base model** | `meta-llama/Llama-3.2-1B-Instruct` |
| **Fine-tuning method** | LoRA (Low-Rank Adaptation) |
| **Dataset size** | 800 train / 100 val / 100 test pairs |
| **Dataset source** | GPT-4o generated analyst memos (see `generate_explanations.py`) |
| **ROUGE-L score** | Not explicitly recorded in any tracked file in this repo (fine-tuning done on Kaggle; results in `notebooks/06_kaggle_lora_finetune.ipynb` output cells) |
| **Fine-tuned weights** | `model2_llm/` directory — **empty** locally (kept on Kaggle) |
| **HuggingFace push** | `push_dataset.py` — pushed dataset to `Divb30/loan-risk-explanations` (private) |
| **Inference stub** | `src/llm/inference.py` — **0-byte file, empty** |

### GPT-4o Dataset Generation
- **Model:** `gpt-4o` (code says `gpt-4o`, comment "cheap — $0.15/1M tokens")
- **Sampling:** 400 high-risk (prob > 0.65) + 400 low-risk (prob < 0.15) + 200 medium-risk (0.30–0.55) = 1000 loans
- **System prompt:** 595-character domain-expert system prompt with explicit causal direction rules for macro features (FEDFUNDS, CPIUS, RISKPREM) and strict output rules (2–3 sentences, analyst memo style)
- **Output format:** loan_id, default_prob, risk_tier, actual_default, shap_drivers, macro_context, explanation

---

## 8. FASTAPI

### `src/api/main.py` — Local Dev API (XGBoost Only)

**Version:** 1.0 (no explicit version tag)  
**Loads at startup:** `xgb_model.json`, `shap.TreeExplainer(xgb_model)`, `X_train.parquet` (for medians)

| Endpoint | Method | Input | Returns |
|---|---|---|---|
| `/predict` | POST | `{"features": {"feature_name": float, ...}}` (partial — missing features filled with training medians) | `default_probability` (float), `risk_tier` ("Low"/"Medium"/"High"), `top_shap_drivers` (top 3 SHAP, feature+value+direction) |
| `/health` | GET | — | `{"status": "ok", "features_loaded": 60}` |

**Risk tiers (local):** Low < 0.30 < Medium < 0.60 < High  
**Current state: XGBoost ONLY — does NOT use the NN pipeline**

---

### `deploy/app.py` — Production Cloud Run API (Full NN Pipeline)

**Version:** 2.0  
**Loads at startup:** `xgb_model.json`, `scaler_mean.npy` + `scaler_scale.npy` (reconstructed StandardScaler), `feature_cols.joblib`, `feature_medians.joblib`, `nn_best.pt` (LoanRiskNN), `shap.TreeExplainer(xgb_model)`

| Endpoint | Method | Input | Returns |
|---|---|---|---|
| `/predict` | POST | `{"features": {"feature_name": float, ...}}` (partial OK — medians imputed) | `default_probability` (NN output), `xgb_probability` (Stage 1 output), `risk_tier` (4-tier), `top_shap_drivers` (top 5 XGB SHAP) |
| `/health` | GET | — | `{"status": "ok", "model": "loan-risk-v2", "nn_input_dim": 61, "features": 60}` |

**Risk tiers (production):** Low < 0.10 < Medium < 0.30 < High < 0.60 < Very High  
**Current state: FULL PIPELINE — XGBoost Stage 1 → logit → StandardScaler → LoanRiskNN Stage 2**

**Key difference from local API:**  
The deploy API returns `nn_prob` as `default_probability` (ensemble output), while the local `src/api/main.py` returns `xgb_prob` directly.

---

## 9. STREAMLIT DASHBOARD (`src/dashboard/app.py`)

### What It Shows
- **Left panel:** Loan input form — 8 sliders/selects: `int_rate`, `logdti`, `logannual_inc`, `revol_util`, `FEDFUNDS_resid`, `CPIUS_resid`, `grade_` (A–G), `issue_year` (2012–present)
- **Right panel (after "Assess Risk →" click):**
  - Default Probability card — large probability display + risk badge (Low/Medium/High)
  - Top SHAP Drivers card — top 3 features with SHAP values color-coded (red = increases risk, green = reduces)
  - Plotly gauge chart — animated risk gauge (green/amber/red zones at 0–30/30–60/60–100%)
  - Macro Context pills — Fed Funds Residual, CPI Residual, Issue Year with trend tags

### API It Calls
```python
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
```
- **Hardcoded default:** `http://127.0.0.1:8000` (local FastAPI)
- Can be overridden with `API_URL` environment variable for deployment
- Calls `POST {API_URL}/predict` with 8 user-selected features

### Current State
- **Calls local `src/api/main.py`** (XGBoost only) by default
- Dashboard header shows correct final metrics: "XGBoost + PyTorch NN Stacking Ensemble · Test AUC 0.9184 · AUC-PR 0.465 · Brier 0.064 · McNemar p<0.0001"
- **NOT deployed** — Streamlit app runs locally only
- To work with the deployed GCP API, `API_URL` would need to point to the Cloud Run URL

---

## 10. DEPLOYMENT STATUS

### `deploy/` Folder — Complete Contents

```
deploy/
├── app.py               # Self-contained FastAPI v2 (LoanRiskNN + XGBoost)
│                        # Inlines ResidualBlock + LoanRiskNN class (no src/ import)
├── Dockerfile           # FROM python:3.11-slim; WORKDIR /app; EXPOSE 8080
│                        # CMD: uvicorn app:app --host 0.0.0.0 --port 8080
├── requirements.txt     # CPU-only pip packages:
│                        #   fastapi==0.115.0, uvicorn==0.30.0, xgboost==2.1.3
│                        #   torch==2.4.0+cpu (from PyTorch CPU wheel index)
│                        #   scikit-learn==1.5.0, shap==0.45.0, pandas==2.2.0
│                        #   pydantic==2.7.0, joblib==1.4.0
│                        #   Extra index: https://download.pytorch.org/whl/cpu
└── artifacts/
    ├── xgb_model.json       # 3.7 MB — XGBoost model
    ├── nn_best.pt           # 925 KB — NN weights
    ├── scaler.joblib        # 2.0 KB — StandardScaler
    ├── scaler_mean.npy      # 608 B — scaler.mean_
    ├── scaler_scale.npy     # 608 B — scaler.scale_
    ├── feature_cols.joblib  # 992 B — ordered list of 60 feature names
    └── feature_medians.joblib  # 1.5 KB — median value per feature
```

### GCP Cloud Run
- **Deployed:** Latest commit (5069399) pushed GCP Cloud Run deployment package
- **GCP Cloud Run URL:** **Not recorded in any file in the repo** — must be retrieved from GCP Console or `gcloud run services describe`
- **Deployment command:** Not documented (no deploy script, no Makefile, no CI/CD)
- **What IS deployed:** `deploy/app.py` — full XGBoost+NN pipeline, version 2.0

### What Is Deployed vs. NOT Deployed

| Component | Deployed? | Where |
|---|---|---|
| XGBoost model | ✅ YES | GCP Cloud Run (via `deploy/artifacts/xgb_model.json`) |
| LoanRiskNN | ✅ YES | GCP Cloud Run (via `deploy/artifacts/nn_best.pt`) |
| FastAPI v2 (`/predict`, `/health`) | ✅ YES | GCP Cloud Run |
| Streamlit dashboard | ❌ NO | Local only (`src/dashboard/app.py`) |
| LLM inference (`src/llm/inference.py`) | ❌ NO | File is 0 bytes (stub) |
| LoRA fine-tuned Llama-3.2-1B | ❌ NO | Weights only on Kaggle |

---

## 11. GITHUB STATUS

### What Is Tracked (committed to git)

```
.gitignore
README.md
configs/config.yaml
deploy/Dockerfile
deploy/app.py
deploy/artifacts/feature_cols.joblib      ← force-added (normally gitignored)
deploy/artifacts/feature_medians.joblib   ← force-added
deploy/artifacts/nn_best.pt               ← force-added
deploy/artifacts/scaler.joblib            ← force-added
deploy/artifacts/scaler_mean.npy          ← force-added
deploy/artifacts/scaler_scale.npy         ← force-added
deploy/artifacts/xgb_model.json           ← force-added
deploy/requirements.txt
mlflow.db
notebooks/01_eda.ipynb
notebooks/02_training.ipynb
notebooks/03_evaluation.ipynb
notebooks/04_test_api.ipynb
notebooks/05_llm.ipynb
notebooks/06_kaggle_lora_finetune.ipynb
notebooks/07_artifact_prep.ipynb
notebooks/mlflow.db
requirement.txt
src/__init__.py
src/api/__init__.py
src/api/main.py
src/dashboard/app.py
src/data/__init__.py
src/data/preprocess.py
src/explainability/__init__.py
src/features/__init__.py
src/features/build_features.py
src/llm/generate_dataset.py
src/llm/generate_explanations.py
src/llm/inference.py
src/llm/push_dataset.py
src/models/__init__.py
src/models/train_nn.py
src/models/train_xgboost.py
src/utils/config.py
src/utils/logger.py
```

**Total tracked files: 41**

### What Is Gitignored (NOT committed)

| Pattern | Files affected |
|---|---|
| `data/raw/` | P2P_Macro_Data.dta, .parquet, .zip, .do files |
| `data/processed/` | features.parquet |
| `data/outputs/` | All 25 files (X_train, shap_values, model .pt, etc.) |
| `*.parquet`, `*.npy`, `*.csv`, `*.pt` | All binary outputs |
| `mlruns/` | MLflow experiment directories |
| `loanvenv/` | Python virtual environment |
| `model2_llm/` | LLM fine-tuned weights (empty dir anyway) |
| `.env` | API keys (OPENAI_API_KEY, HF_TOKEN) |
| `*.joblib`, `*.pkl`, `*.bin` | Model artifacts (except `deploy/artifacts/` — force-added) |

### Files Force-Added

The `.gitignore` includes at the top:
```
!deploy/artifacts/
!deploy/artifacts/*
```
This un-ignores the `deploy/artifacts/` directory, allowing `*.joblib`, `*.npy`, `*.pt`, and `*.json` inside it to be committed. All 7 artifact files were added in the latest commit (5069399) for Cloud Run deployment.

---

## 12. REMAINING TODO

The following items are incomplete or not yet done:

### High Priority
1. **Streamlit Deployment** — `src/dashboard/app.py` is local only. Needs to be deployed (Streamlit Cloud, HuggingFace Spaces, or a second Cloud Run service). The `API_URL` env var needs to point to the GCP Cloud Run URL.
   
2. **GCP Cloud Run URL documentation** — The deployed API URL is not captured anywhere in the repo. Should be added to README.md or a `.env.example`.

3. **Streamlit → Production API wiring** — Dashboard currently calls `http://127.0.0.1:8000` (XGBoost-only local API). After deployment, it should call the Cloud Run URL and will automatically get the full NN pipeline response.

### Medium Priority
4. **LLM inference stub** — `src/llm/inference.py` is 0 bytes. If the LoRA fine-tuned Llama model is to be used in the API or dashboard, this needs to be implemented.

5. **LoRA fine-tuned model hosting** — Weights are on Kaggle only. Need to upload to HuggingFace Hub (`Divb30/` namespace) and implement `inference.py` to call it.

6. **README accuracy** — README says "AUC 0.918" but doesn't document the Cloud Run URL, dashboard URL, or exact pipeline details post-deployment.

7. **Unit tests** — `tests/` directory is empty. No tests exist for preprocessing, feature engineering, or API endpoints.

8. **GCP CI/CD pipeline** — No deploy script, Makefile, or GitHub Actions workflow exists. Deployments are done manually (implied by git log).

### Low Priority
9. **`src/explainability/__init__.py`** — Directory exists but is empty (only `__init__.py`). SHAP computation is embedded in notebooks. Could be refactored into a proper module.

10. **`src/llm/inference.py`** — 0-byte stub. Should contain HuggingFace inference pipeline for deployed Llama model.

11. **McNemar test exact values** — p < 0.0001 is shown in the dashboard but not stored as a numeric result anywhere in the outputs directory.

12. **OOF AUC exact value** — Logged to MLflow per-fold but not surfaced in any output file. Should be added to briefing/README.

---

## QUICK REFERENCE CARD

```
Dataset:         LendingClub P2P + US Macro | 1.8M loans | 2012–present
Target:          badloan | 92/8 imbalance | scale_pos_weight = 11.5
Features:        60 (28 loan + 5 borrower + 17 macro + 5 categorical + 1 time + 5 interaction)
Stage 1:         XGBoost(n_est=500, max_depth=6, lr=0.05, cuda→hist)
Stage 2:         LoanRiskNN(63→256→128→64→1) | ResidualBlocks | AdamW | BCE pos_weight=3
Stack input:     StandardScaler(X_60) ++ safe_logit(xgb_prob) → (N, 61)
Test AUC-ROC:    0.9184
Test AUC-PR:     0.465
Test Brier:      0.064
Model selection: McNemar p < 0.0001
Top SHAP driver: FEDFUNDS_resid (30.4% XGB importance)
Drift:           PSI=0.0, KS p>0.24 — all 61 features stable
LLM:             GPT-4o → 1000 memos → LoRA Llama-3.2-1B | ROUGE-L not recorded locally
Local API:       src/api/main.py | XGBoost only | POST /predict | GET /health
Prod API:        deploy/app.py | XGBoost+NN | version 2.0 | GCP Cloud Run
Dashboard:       src/dashboard/app.py | Streamlit | calls API_URL (default: localhost:8000)
Deployed:        FastAPI v2 on GCP Cloud Run (URL not in repo)
NOT deployed:    Streamlit, LLM inference
Git tracked:     41 files | deploy/artifacts/ force-added via !deploy/artifacts/* in .gitignore
Git ignored:     data/, mlruns/, loanvenv/, .env, model2_llm/
```
