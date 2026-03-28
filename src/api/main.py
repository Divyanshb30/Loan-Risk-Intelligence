from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import shap
import joblib
from pathlib import Path

from src.utils.config import load_config, get_project_root
from src.models.train_nn import LoanRiskNN

app = FastAPI(title="Loan Risk Intelligence API", version="2.0")
config = load_config()
output_dir = Path(get_project_root()) / config["paths"]["outputs"]
nn_dropout = config["nn"]["dropout"]

# ── Load artifacts once at startup ─────────────────────────────
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(output_dir / "xgb_model.json"))
explainer = shap.TreeExplainer(xgb_model)

X_train = pd.read_parquet(output_dir / "X_train.parquet")
FEATURE_COLS = X_train.columns.tolist()
MEDIANS = X_train.median().to_dict()

scaler = joblib.load(output_dir / "scaler.joblib")

nn_model = LoanRiskNN(input_dim=len(FEATURE_COLS) + 1, dropout=nn_dropout)
nn_model.load_state_dict(
    torch.load(output_dir / "nn_best.pt", map_location="cpu", weights_only=True)
)
nn_model.eval()


class LoanFeatures(BaseModel):
    features: Dict[str, float]


class ShapDriver(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    default_probability: float
    xgb_probability: float
    risk_tier: str
    top_shap_drivers: List[ShapDriver]


def get_risk_tier(p: float) -> str:
    if p < 0.10:
        return "Low"
    if p < 0.30:
        return "Medium"
    if p < 0.60:
        return "High"
    return "Very High"


@app.post("/predict", response_model=PredictionResponse)
def predict(loan: LoanFeatures):
    try:
        user = dict(loan.features)
        # int_rate: training uses APR as a fraction (e.g. 0.125 = 12.5%).
        if "int_rate" in user and user["int_rate"] > 1.0:
            user["int_rate"] = user["int_rate"] / 100.0

        # revol_util: training uses 0–1; UIs often send 0–100.
        if "revol_util" in user and user["revol_util"] > 1.0:
            user["revol_util"] = user["revol_util"] / 100.0

        full_features = {**MEDIANS, **user}
        row = pd.DataFrame([full_features]).reindex(columns=FEATURE_COLS)

        xgb_prob = float(xgb_model.predict_proba(row)[:, 1][0])

        sv = explainer(row).values[0]
        top_idx = np.argsort(np.abs(sv))[::-1][:5]
        shap_drivers = [
            ShapDriver(
                feature=FEATURE_COLS[i],
                shap_value=round(float(sv[i]), 4),
                direction="increases_risk" if sv[i] > 0 else "reduces_risk",
            )
            for i in top_idx
        ]

        eps = 1e-7
        xgb_logit = float(
            np.log(
                np.clip(xgb_prob, eps, 1 - eps)
                / (1 - np.clip(xgb_prob, eps, 1 - eps))
            )
        )
        x_scaled = scaler.transform(row.values)
        x_nn = np.hstack([x_scaled, [[xgb_logit]]]).astype(np.float32)

        with torch.no_grad():
            nn_prob = torch.sigmoid(nn_model(torch.tensor(x_nn))).item()

        return PredictionResponse(
            default_probability=round(nn_prob, 4),
            xgb_probability=round(xgb_prob, 4),
            risk_tier=get_risk_tier(nn_prob),
            top_shap_drivers=shap_drivers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "loan-risk-v2",
        "nn_input_dim": len(FEATURE_COLS) + 1,
        "features": len(FEATURE_COLS),
    }
