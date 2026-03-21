from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
import shap
import joblib
import os
from pathlib import Path
from src.utils.config import load_config, get_project_root

app    = FastAPI(title="Loan Risk Intelligence API")
config = load_config()
output_dir = Path(get_project_root()) / config["paths"]["outputs"]

# ── Load everything once at startup ──────────────────────────
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(output_dir / "xgb_model.json"))
explainer  = shap.TreeExplainer(xgb_model)

X_train        = pd.read_parquet(output_dir / "X_train.parquet")
FEATURE_MEDIANS = X_train.median().to_dict()
feature_names  = X_train.columns.tolist()

class LoanFeatures(BaseModel):
    features: dict[str, float]

@app.post("/predict")
def predict(loan: LoanFeatures):
    # Fill all 60 features with training medians, override with whatever was passed in
    full_features = {**FEATURE_MEDIANS, **loan.features}
    row = pd.DataFrame([full_features]).reindex(columns=feature_names)

    prob      = float(xgb_model.predict_proba(row)[:, 1][0])
    risk_tier = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

    sv      = explainer(row).values[0]
    top_idx = np.argsort(np.abs(sv))[::-1][:3]
    shap_drivers = [
        {
            "feature":    feature_names[i],
            "shap_value": round(float(sv[i]), 4),
            "direction":  "increases_risk" if sv[i] > 0 else "reduces_risk"
        }
        for i in top_idx
    ]

    return {
        "default_probability": round(prob, 4),
        "risk_tier":           risk_tier,
        "top_shap_drivers":    shap_drivers
    }

@app.get("/health")
def health():
    return {"status": "ok", "features_loaded": len(feature_names)}
