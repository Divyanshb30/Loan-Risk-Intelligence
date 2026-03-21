import numpy as np
import pandas as pd
import json
from pathlib import Path
from src.utils.config import load_config, get_project_root

config     = load_config()
output_dir = Path(get_project_root()) / config["paths"]["outputs"]

# ── Load what we already have ─────────────────────────────────
X_test      = pd.read_parquet(output_dir / "X_test.parquet")
y_test      = pd.read_parquet(output_dir / "y_test.parquet").squeeze().values
shap_values = np.load(output_dir / "shap_values.npy")
nn_probs    = pd.read_parquet(output_dir / "nn_predictions.parquet")["y_prob_nn"].values

feature_names = X_test.columns.tolist()

# ── Sample 500 diverse loans ──────────────────────────────────
# 200 high risk, 200 low risk, 100 medium — covers the full range
np.random.seed(42)

high_idx = np.where(nn_probs > 0.65)[0]
low_idx  = np.where(nn_probs < 0.15)[0]
mid_idx  = np.where((nn_probs >= 0.3) & (nn_probs <= 0.55))[0]

sampled = np.concatenate([
    np.random.choice(high_idx, min(400, len(high_idx)), replace=False),
    np.random.choice(low_idx,  min(400, len(low_idx)),  replace=False),
    np.random.choice(mid_idx,  min(200, len(mid_idx)),  replace=False),
])

# ── Build structured records ──────────────────────────────────
records = []
MACRO_FEATURES = [
    'FEDFUNDS_resid', 'CPIUS_resid', 'inf_resid',
    'riskprem_resid', 'FED_lag6_resid', 'muni_6m_resid'
]

for idx in sampled:
    prob = float(nn_probs[idx])
    sv   = shap_values[idx]
    row  = X_test.iloc[idx]

    # Top 5 SHAP drivers
    top5_idx = np.argsort(np.abs(sv))[::-1][:5]
    drivers  = [
        {
            "feature":    feature_names[i],
            "shap_value": round(float(sv[i]), 4),
            "direction":  "increases_risk" if sv[i] > 0 else "reduces_risk",
            "raw_value":  round(float(row[feature_names[i]]), 4)
        }
        for i in top5_idx
    ]

    # Macro snapshot
    macro = {
        f: round(float(row[f]), 4)
        for f in MACRO_FEATURES if f in row.index
    }

    risk_tier = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

    records.append({
        "loan_id":         int(idx),
        "default_prob":    round(prob, 4),
        "risk_tier":       risk_tier,
        "actual_default":  int(y_test[idx]),
        "shap_drivers":    drivers,
        "macro_context":   macro,
        "issue_year":      int(row.get("issue_year", 0)),
        "int_rate":        round(float(row.get("int_rate", 0)), 4),
        "grade":           int(row.get("grade_", 0)),
    })

# ── Save ──────────────────────────────────────────────────────
out_path = output_dir / "shap_dataset_raw.json"
with open(out_path, "w") as f:
    json.dump(records, f, indent=2)

print(f"Saved {len(records)} records to {out_path}")
print(f"High: {sum(1 for r in records if r['risk_tier']=='High')} | "
      f"Medium: {sum(1 for r in records if r['risk_tier']=='Medium')} | "
      f"Low: {sum(1 for r in records if r['risk_tier']=='Low')}")

# ── Also save as CSV for quick inspection ─────────────────────
rows_flat = []
for r in records:
    rows_flat.append({
        "loan_id":       r["loan_id"],
        "default_prob":  r["default_prob"],
        "risk_tier":     r["risk_tier"],
        "actual_default":r["actual_default"],
        "driver_1":      r["shap_drivers"][0]["feature"],
        "shap_1":        r["shap_drivers"][0]["shap_value"],
        "driver_2":      r["shap_drivers"][1]["feature"],
        "shap_2":        r["shap_drivers"][1]["shap_value"],
        "driver_3":      r["shap_drivers"][2]["feature"],
        "shap_3":        r["shap_drivers"][2]["shap_value"],
        "FEDFUNDS_resid":r["macro_context"].get("FEDFUNDS_resid", 0),
        "CPIUS_resid":   r["macro_context"].get("CPIUS_resid", 0),
        "issue_year":    r["issue_year"],
    })

pd.DataFrame(rows_flat).to_csv(output_dir / "shap_dataset_raw.csv", index=False)
print("CSV preview saved.")
