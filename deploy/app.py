from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler


app  = FastAPI(title="Loan Risk Intelligence API", version="2.0")
BASE = Path(__file__).parent / "artifacts"


# ── Architecture (inlined — no src/ import needed in Docker) ─────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(x + self.block(x)))


class LoanRiskNN(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout)
        )
        self.res_block1 = ResidualBlock(256, dropout)   # was resblock1
        self.compress1  = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout)
        )
        self.res_block2 = ResidualBlock(128, dropout)   # was resblock2
        self.compress2  = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout)
        )
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = self.res_block1(x)   # was resblock1
        x = self.compress1(x)
        x = self.res_block2(x)   # was resblock2
        x = self.compress2(x)
        return self.output(x).squeeze(1)



# ── Load Artifacts Once at Startup ───────────────────────────────────────────

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(BASE / "xgb_model.json"))


scaler       = StandardScaler()
scaler.mean_ = np.load(BASE / "scaler_mean.npy")
scaler.scale_= np.load(BASE / "scaler_scale.npy")
scaler.var_  = scaler.scale_ ** 2
scaler.n_features_in_ = len(scaler.mean_)
FEATURE_COLS = joblib.load(BASE / "feature_cols.joblib")   # e.g. 60 or 62 features
MEDIANS      = joblib.load(BASE / "feature_medians.joblib")

# NN input_dim = feature count + 1 (XGB logit appended — matches training pipeline)
nn_model = LoanRiskNN(input_dim=len(FEATURE_COLS) + 1)
nn_model.load_state_dict(
    torch.load(BASE / "nn_best.pt", map_location="cpu", weights_only=True)
)
nn_model.eval()

explainer = shap.TreeExplainer(xgb_model)


# ── Schema ───────────────────────────────────────────────────────────────────

class LoanFeatures(BaseModel):
    features: Dict[str, float]

class ShapDriver(BaseModel):
    feature:    str
    shap_value: float
    direction:  str

class PredictionResponse(BaseModel):
    default_probability: float
    xgb_probability:     float
    risk_tier:           str
    top_shap_drivers:    List[ShapDriver]


def get_risk_tier(p: float) -> str:
    if p < 0.10: return "Low"
    if p < 0.30: return "Medium"
    if p < 0.60: return "High"
    return "Very High"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
def predict(loan: LoanFeatures):
    try:
        # Fill missing features with training medians, then align column order
        full_features = {**MEDIANS, **loan.features}
        row = pd.DataFrame([full_features]).reindex(columns=FEATURE_COLS)

        # ── Stage 1: XGBoost ─────────────────────────────────────────────────
        xgb_prob = float(xgb_model.predict_proba(row)[:, 1][0])

        # SHAP from XGBoost (interpretable feature drivers)
        sv      = explainer(row).values[0]
        top_idx = np.argsort(np.abs(sv))[::-1][:5]
        shap_drivers = [
            ShapDriver(
                feature    = FEATURE_COLS[i],
                shap_value = round(float(sv[i]), 4),
                direction  = "increases_risk" if sv[i] > 0 else "reduces_risk"
            )
            for i in top_idx
        ]

        # ── Stage 2: NN (replicates training pipeline exactly) ───────────────
        eps        = 1e-7
        xgb_logit  = float(np.log(
            np.clip(xgb_prob, eps, 1 - eps) / (1 - np.clip(xgb_prob, eps, 1 - eps))
        ))
        x_scaled   = scaler.transform(row.values)                         # (1, n_features)
        x_nn       = np.hstack([x_scaled, [[xgb_logit]]]).astype(np.float32)  # (1, n_features+1)

        with torch.no_grad():
            nn_prob = torch.sigmoid(nn_model(torch.tensor(x_nn))).item()

        # NN is the ensemble output — XGB logit is baked into its input
        return PredictionResponse(
            default_probability = round(nn_prob, 4),
            xgb_probability     = round(xgb_prob, 4),
            risk_tier           = get_risk_tier(nn_prob),
            top_shap_drivers    = shap_drivers,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        "loan-risk-v2",
        "nn_input_dim": len(FEATURE_COLS) + 1,
        "features":     len(FEATURE_COLS),
    }
