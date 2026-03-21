import streamlit as st
import requests
import plotly.graph_objects as go
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────────
API_URL      = os.getenv("API_URL", "http://127.0.0.1:8000")
CURRENT_YEAR = datetime.now().year
YEAR_OPTIONS = list(range(2012, CURRENT_YEAR + 1))

st.set_page_config(
    page_title="Loan Risk Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .stApp { background: #0a0a0f; color: #e2e8f0; }
    section[data-testid="stSidebar"] { display: none; }
    .main .block-container { padding: 2rem 3rem; max-width: 1400px; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    .header-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #1e3a5f; border-radius: 16px;
        padding: 2rem 2.5rem; margin-bottom: 2rem;
        display: flex; align-items: center; justify-content: space-between;
    }
    .header-title { font-size: 1.8rem; font-weight: 700; color: #fff; margin: 0; }
    .header-subtitle { font-size: 0.85rem; color: #64748b; margin-top: 0.3rem; }
    .header-badge {
        background: #0f3460; border: 1px solid #1e4d8c; border-radius: 8px;
        padding: 0.5rem 1rem; font-size: 0.8rem; color: #60a5fa; font-weight: 500;
    }

    .card {
        background: #111827; border: 1px solid #1f2937;
        border-radius: 12px; padding: 1.5rem; height: 100%;
    }
    .card-title {
        font-size: 0.75rem; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;
    }

    .stSlider > div > div > div > div { background: #3b82f6 !important; }
    div[data-baseweb="select"] > div {
        background: #1f2937 !important; border-color: #374151 !important;
        color: #e2e8f0 !important; border-radius: 8px !important;
    }
    .stSlider label, div[data-testid="stSlider"] label {
        color: #9ca3af !important; font-size: 0.8rem !important;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; padding: 0.75rem 2rem !important;
        font-weight: 600 !important; font-size: 0.95rem !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(59,130,246,0.5) !important;
    }

    .risk-badge-high {
        display: inline-block; background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.4); color: #ef4444;
        padding: 0.4rem 1.2rem; border-radius: 999px;
        font-weight: 700; font-size: 1rem; letter-spacing: 0.05em;
    }
    .risk-badge-medium {
        display: inline-block; background: rgba(245,158,11,0.15);
        border: 1px solid rgba(245,158,11,0.4); color: #f59e0b;
        padding: 0.4rem 1.2rem; border-radius: 999px;
        font-weight: 700; font-size: 1rem; letter-spacing: 0.05em;
    }
    .risk-badge-low {
        display: inline-block; background: rgba(16,185,129,0.15);
        border: 1px solid rgba(16,185,129,0.4); color: #10b981;
        padding: 0.4rem 1.2rem; border-radius: 999px;
        font-weight: 700; font-size: 1rem; letter-spacing: 0.05em;
    }

    .driver-row {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.6rem 0; border-bottom: 1px solid #1f2937;
    }
    .driver-name { font-size: 0.85rem; color: #d1d5db; font-weight: 500; }
    .driver-val-pos {
        font-size: 0.85rem; font-weight: 600; color: #ef4444;
        background: rgba(239,68,68,0.1); padding: 0.2rem 0.6rem; border-radius: 6px;
    }
    .driver-val-neg {
        font-size: 0.85rem; font-weight: 600; color: #10b981;
        background: rgba(16,185,129,0.1); padding: 0.2rem 0.6rem; border-radius: 6px;
    }

    .macro-pill {
        background: #1f2937; border: 1px solid #374151;
        border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
    }
    .macro-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; }
    .macro-value { font-size: 1.4rem; font-weight: 700; color: #f9fafb; margin: 0.3rem 0; }
    .macro-tag-up   { font-size: 0.75rem; color: #ef4444; }
    .macro-tag-down { font-size: 0.75rem; color: #10b981; }

    .section-label {
        font-size: 0.7rem; font-weight: 600; color: #4b5563;
        text-transform: uppercase; letter-spacing: 0.12em; margin: 1.5rem 0 0.8rem 0;
    }
    .prob-display {
        font-size: 4rem; font-weight: 700;
        line-height: 1; letter-spacing: -0.02em;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div>
        <div class="header-title"> Loan Risk Intelligence</div>
        <div class="header-subtitle">XGBoost + PyTorch NN Stacking Ensemble · 1.8M Lending Club loans</div>
    </div>
    <div style="display:flex; gap:0.75rem; flex-wrap:wrap;">
        <div class="header-badge">AUC 0.9184</div>
        <div class="header-badge">AUC-PR 0.465</div>
        <div class="header-badge">Brier 0.064</div>
        <div class="header-badge">McNemar p&lt;0.0001</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Loan Parameters</div>', unsafe_allow_html=True)

    int_rate      = st.slider("Interest Rate",             0.05,  0.35,  0.15, 0.01, format="%.2f")
    logdti        = st.slider("Log DTI",                   0.0,   4.0,   2.1,  0.1)
    logannual_inc = st.slider("Log Annual Income",         8.0,   13.0,  10.5, 0.1)
    revol_util    = st.slider("Revolving Utilisation (%)", 0.0,   100.0, 50.0, 1.0)

    st.markdown('<div class="section-label">Macro Environment</div>', unsafe_allow_html=True)
    FEDFUNDS_resid = st.slider("Fed Funds Residual", -2.0, 2.0, 0.0, 0.05)
    CPIUS_resid    = st.slider("CPI Residual",       -2.0, 2.0, 0.0, 0.05)

    st.markdown('<div class="section-label">Loan Profile</div>', unsafe_allow_html=True)
    grade_ = st.selectbox(
        "Grade", [1,2,3,4,5,6,7],
        format_func=lambda x: ["A","B","C","D","E","F","G"][x-1]
    )
    issue_year = st.selectbox(
        "Issue Year", YEAR_OPTIONS,
        index=YEAR_OPTIONS.index(CURRENT_YEAR)
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Assess Risk →", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────
with right:
    if predict_btn:
        features = {
            "int_rate":       int_rate,
            "logdti":         logdti,
            "logannual_inc":  logannual_inc,
            "grade_":         float(grade_),
            "revol_util":     revol_util,
            "FEDFUNDS_resid": FEDFUNDS_resid,
            "CPIUS_resid":    CPIUS_resid,
            "issue_year":     float(issue_year)
        }

        try:
            with st.spinner("Assessing risk..."):
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"features": features},
                    timeout=10
                )
                resp.raise_for_status()
                result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("⚠️ API unavailable — ensure the FastAPI server is running: `uvicorn src.api.main:app --reload`")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("⚠️ Request timed out. Try again.")
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        prob  = result["default_probability"]
        tier  = result["risk_tier"]
        shaps = result["top_shap_drivers"]

        badge_class = {"High": "risk-badge-high", "Medium": "risk-badge-medium", "Low": "risk-badge-low"}[tier]
        prob_color  = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}[tier]

        r1, r2 = st.columns([1, 1], gap="medium")

        # ── Probability card ──────────────────────────────────
        with r1:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:2rem;">
                <div class="card-title">Default Probability</div>
                <div class="prob-display" style="color:{prob_color};">{prob:.1%}</div>
                <br>
                <span class="{badge_class}">{tier} Risk</span>
                <br><br>
                <div style="font-size:0.8rem; color:#6b7280;">
                    Low &lt;30% &nbsp;·&nbsp; Medium 30–60% &nbsp;·&nbsp; High &gt;60%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── SHAP drivers card ─────────────────────────────────
        with r2:
            drivers_html = ""
            for d in shaps:
                val_class = "driver-val-pos" if d["direction"] == "increases_risk" else "driver-val-neg"
                sign = "+" if d["direction"] == "increases_risk" else ""
                drivers_html += f"""
                <div class="driver-row">
                    <span class="driver-name">{d['feature']}</span>
                    <span class="{val_class}">{sign}{d['shap_value']:.4f}</span>
                </div>"""

            st.markdown(f"""
            <div class="card">
                <div class="card-title">Top SHAP Drivers</div>
                {drivers_html}
                <div style="font-size:0.72rem; color:#4b5563; margin-top:0.8rem;">
                    🔴 increases default risk &nbsp; 🟢 reduces default risk
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Gauge ─────────────────────────────────────────────
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"valueformat": ".1%", "font": {"color": prob_color, "size": 36}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#4b5563",
                         "tickfont": {"color": "#4b5563", "size": 11}},
                "bar":  {"color": prob_color, "thickness": 0.25},
                "bgcolor": "#111827", "bordercolor": "#1f2937",
                "steps": [
                    {"range": [0.0, 0.3], "color": "rgba(16,185,129,0.15)"},
                    {"range": [0.3, 0.6], "color": "rgba(245,158,11,0.12)"},
                    {"range": [0.6, 1.0], "color": "rgba(239,68,68,0.15)"},
                ],
            }
        ))
        fig.update_layout(
            height=220, margin=dict(t=20, b=0, l=20, r=20),
            paper_bgcolor="#111827", font_color="#e2e8f0"
        )
        st.markdown('<div class="card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Risk Gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Macro context ──────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Macro Context</div>',
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        macro_items = [
            (m1, "Fed Funds Residual", FEDFUNDS_resid, f"{FEDFUNDS_resid:+.3f}", True),
            (m2, "CPI Residual",       CPIUS_resid,    f"{CPIUS_resid:+.3f}",    True),
            (m3, "Issue Year",         issue_year,     str(issue_year),           False),
        ]
        for col, label, val, fmt, is_float in macro_items:
            if is_float:
                tag_class = "macro-tag-up"   if val > 0 else "macro-tag-down"
                tag_text  = "above trend 🔴" if val > 0 else "below trend 🟢"
            else:
                tag_class = "macro-tag-up"   if val > 2016 else "macro-tag-down"
                tag_text  = "post-2016 regime" if val > 2016 else "pre-2016 regime"

            col.markdown(f"""
            <div class="macro-pill">
                <div class="macro-label">{label}</div>
                <div class="macro-value">{fmt}</div>
                <div class="{tag_class}">{tag_text}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:4rem 2rem; min-height:400px;
             display:flex; flex-direction:column; align-items:center; justify-content:center;">
            <div style="font-size:3rem; margin-bottom:1rem;">🔍</div>
            <div style="font-size:1.1rem; color:#6b7280; font-weight:500;">
                Configure loan parameters and click<br>
                <span style="color:#3b82f6;">Assess Risk →</span> to generate a prediction
            </div>
            <div style="margin-top:2rem; font-size:0.8rem; color:#374151;">
                Model: XGBoost + PyTorch NN &nbsp;·&nbsp; Test AUC 0.9184 &nbsp;·&nbsp; McNemar p&lt;0.0001
            </div>
        </div>
        """, unsafe_allow_html=True)
