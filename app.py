import streamlit as st
import joblib
import numpy as np
import pandas as pd
import gdown
import os
from pathlib import Path

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #08090c;
    color: #e8e8e8;
}

.stApp { background: #08090c; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 48px 64px;
    max-width: 1200px;
}

/* ── HEADER ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 48px;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 48px;
}

.logo {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    color: #ffffff;
    letter-spacing: 2px;
}

.logo span { color: #3b82f6; }

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0f1a0f;
    border: 1px solid #166534;
    border-radius: 100px;
    padding: 6px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #4ade80;
}

.status-dot {
    width: 6px;
    height: 6px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── HERO ── */
.hero-title {
    font-size: 3.5rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
    letter-spacing: -2px;
    margin: 0;
}

.hero-title .accent { color: #3b82f6; }

.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    margin-top: 16px;
    font-weight: 300;
    line-height: 1.6;
    max-width: 500px;
}

/* ── METRICS ROW ── */
.metrics-row {
    display: flex;
    gap: 1px;
    background: #1a1a1a;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    overflow: hidden;
    margin: 48px 0;
}

.metric-item {
    flex: 1;
    background: #0d0d0d;
    padding: 20px 24px;
}

.metric-item:hover { background: #111111; }

.metric-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4b5563;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #ffffff;
}

.metric-val.blue { color: #3b82f6; }
.metric-val.green { color: #4ade80; }
.metric-val.amber { color: #f59e0b; }

/* ── SECTION ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4b5563;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* ── INPUTS ── */
div[data-testid="stNumberInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #6b7280 !important;
    letter-spacing: 1px !important;
}

div[data-testid="stNumberInput"] input {
    background: #0d0d0d !important;
    color: #e8e8e8 !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

div[data-testid="stNumberInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1) !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: #3b82f6 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 14px 32px !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: background 0.15s !important;
}

.stButton > button:hover {
    background: #2563eb !important;
}

/* ── RESULT ── */
.result-fraud {
    border: 1px solid #991b1b;
    background: #0c0404;
    border-radius: 12px;
    padding: 40px 48px;
}

.result-legit {
    border: 1px solid #166534;
    background: #040c04;
    border-radius: 12px;
    padding: 40px 48px;
}

.result-verdict {
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin: 0;
}

.result-desc {
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 8px;
}

.result-prob {
    font-family: 'DM Mono', monospace;
    font-size: 3rem;
    font-weight: 500;
    margin: 0;
}

.result-prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4b5563;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

.divider {
    border: none;
    border-top: 1px solid #1a1a1a;
    margin: 40px 0;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 1px solid #1a1a1a !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #4b5563 !important;
    padding: 12px 20px !important;
    border-bottom: 2px solid transparent !important;
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding: 24px 0 !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ───────────────────────────────────────────
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FILES = {
    "models/final_model.joblib":    "1axYmDQ0i6_qNBKOcV2rz169gGJDYloOF",
    "models/scaler.joblib":         "1jz20q91NGwtobMoh99tO8XmxczuPyG8F",
    "models/best_threshold.joblib": "1Ni1zKQwOVjCdNIMPHWFePXe5cP2-EBbN",
}

@st.cache_resource
def load_models():
    for path, file_id in FILES.items():
        if not os.path.exists(path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=True)
    model     = joblib.load("models/final_model.joblib")
    scaler    = joblib.load("models/scaler.joblib")
    threshold = joblib.load("models/best_threshold.joblib")
    return model, scaler, threshold

model, scaler, threshold = load_models()

FRAUD_EXAMPLE = {
    'Time': 406.0, 'V1': -2.3122, 'V2': 1.9520, 'V3': -1.6099,
    'V4': 3.9979, 'V5': -0.5222, 'V6': -1.4265, 'V7': -2.5374,
    'V8': 1.3917, 'V9': -2.7701, 'V10': -2.7723, 'V11': 3.2020,
    'V12': -2.8999, 'V13': -0.5952, 'V14': -4.2893, 'V15': 0.3897,
    'V16': -1.1407, 'V17': -2.8301, 'V18': -0.0168, 'V19': 0.4170,
    'V20': 0.1269, 'V21': 0.5172, 'V22': -0.0350, 'V23': -0.4652,
    'V24': 0.3202, 'V25': 0.0, 'V26': 0.0, 'V27': 0.2611,
    'V28': -0.1433, 'Amount': 0.0
}

# ── TOP BAR ───────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <span class="logo">FRAUD<span>DETECTION</span></span>
    <span class="status-badge">
        <span class="status-dot"></span>
        MODEL ACTIVE
    </span>
</div>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────
with col_hero:
    st.markdown("""
    <p class="hero-title">
        Détectez la fraude <span class="accent">avant qu'elle ne se produise.</span>
    </p>

    <p class="hero-sub">
        Real-time transaction scoring powered by Extra Trees Classifier.
        <br>
        Trained on 284,807 European card transactions.
    </p>
    """, unsafe_allow_html=True)

# ── METRICS ───────────────────────────────────────────────
st.markdown(f"""
<div class="metrics-row">
    <div class="metric-item">
        <div class="metric-name">PR-AUC</div>
        <div class="metric-val blue">0.8808</div>
    </div>
    <div class="metric-item">
        <div class="metric-name">Precision</div>
        <div class="metric-val green">93.26%</div>
    </div>
    <div class="metric-item">
        <div class="metric-name">Recall</div>
        <div class="metric-val green">84.69%</div>
    </div>
    <div class="metric-item">
        <div class="metric-name">F1 Score</div>
        <div class="metric-val">0.8877</div>
    </div>
    <div class="metric-item">
        <div class="metric-name">Threshold</div>
        <div class="metric-val amber">{threshold:.4f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── INPUTS ────────────────────────────────────────────────
st.markdown('<p class="section-label">Transaction Data</p>', unsafe_allow_html=True)

load_fraud = st.button("Load fraud example →")

col1, col2 = st.columns(2)
with col1:
    time_val = st.number_input("TIME", value=float(FRAUD_EXAMPLE['Time']) if load_fraud else 0.0, format="%.2f")
with col2:
    amount_val = st.number_input("AMOUNT (€)", value=float(FRAUD_EXAMPLE['Amount']) if load_fraud else 0.0, min_value=0.0, format="%.2f")

st.markdown('<p class="section-label" style="margin-top:24px;">PCA Components</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["V1 — V7", "V8 — V14", "V15 — V21", "V22 — V28"])

v_values = {}
for tab, (start, end) in zip([tab1, tab2, tab3, tab4], [(1,8),(8,15),(15,22),(22,29)]):
    with tab:
        cols = st.columns(7)
        for i, col in zip(range(start, end), cols):
            with col:
                v_values[f'V{i}'] = st.number_input(
                    f"V{i}",
                    value=float(FRAUD_EXAMPLE[f'V{i}']) if load_fraud else 0.0,
                    format="%.4f", key=f"v{i}"
                )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

analyze = st.button("Analyze Transaction")

# ── RESULT ────────────────────────────────────────────────
if analyze:
    features = {'Time': time_val, **{f'V{i}': v_values[f'V{i}'] for i in range(1, 29)}, 'Amount': amount_val}
    input_df = pd.DataFrame([features])
    input_scaled = input_df.copy()
    input_scaled[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])
    proba = model.predict_proba(input_scaled)[0][1]
    prediction = int(proba >= threshold)

    st.markdown('<p class="section-label" style="margin-top:32px;">Analysis Result</p>', unsafe_allow_html=True)

    col_verdict, col_prob = st.columns([3, 1])

    with col_verdict:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-fraud">
                <p class="result-verdict" style="color:#ef4444;">⚠ Fraudulent Transaction</p>
                <p class="result-desc">This transaction has been flagged as suspicious. Confidence: {proba:.2%}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
                <p class="result-verdict" style="color:#4ade80;">✓ Legitimate Transaction</p>
                <p class="result-desc">This transaction appears safe. Confidence: {1-proba:.2%}</p>
            </div>""", unsafe_allow_html=True)

    with col_prob:
        color = "#ef4444" if prediction == 1 else "#4ade80"
        st.markdown(f"""
        <div style="text-align:center; padding: 40px 0;">
            <p class="result-prob" style="color:{color};">{proba:.1%}</p>
            <p class="result-prob-label">Fraud probability</p>
        </div>""", unsafe_allow_html=True)

    st.progress(float(proba))

    with st.expander("View input data"):
        st.dataframe(input_df.style.format("{:.4f}"), use_container_width=True)