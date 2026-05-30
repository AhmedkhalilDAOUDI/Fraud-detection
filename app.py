import streamlit as st
import joblib
import numpy as np
import pandas as pd
import gdown
import os
from pathlib import Path

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

* { font-family: 'Syne', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }

.stApp {
    background: #070B14;
}

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1f2937;
}

.header-container {
    background: linear-gradient(135deg, #070B14 0%, #0d1520 50%, #070B14 100%);
    border: 1px solid #1a3a5c;
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}

.header-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(0, 200, 255, 0.03) 0%, transparent 60%);
}

.header-title {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1px;
}

.header-title span { color: #00c8ff; }

.header-subtitle {
    color: #6b7280;
    font-size: 1rem;
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 100px;
    padding: 8px 16px;
    margin: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #9ca3af;
}

.metric-pill .value { color: #00c8ff; font-weight: 700; }

.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: #4b5563;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 16px;
    margin-top: 32px;
}

.input-grid {
    background: #0d1117;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px;
}

.fraud-result {
    background: linear-gradient(135deg, #1a0000 0%, #2d0000 100%);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    box-shadow: 0 0 60px rgba(239, 68, 68, 0.15), inset 0 0 60px rgba(239, 68, 68, 0.05);
}

.legit-result {
    background: linear-gradient(135deg, #001a0d 0%, #002d1a 100%);
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    box-shadow: 0 0 60px rgba(16, 185, 129, 0.15), inset 0 0 60px rgba(16, 185, 129, 0.05);
}

.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -2px;
}

.result-sub {
    font-size: 0.9rem;
    margin-top: 8px;
    opacity: 0.7;
}

.prob-display {
    background: #0d1117;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.prob-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    color: #f59e0b;
    margin: 0;
}

.prob-label {
    font-size: 0.75rem;
    color: #4b5563;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

.example-btn {
    background: #111827;
    border: 1px dashed #374151;
    border-radius: 8px;
    padding: 12px 24px;
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
}

div[data-testid="stNumberInput"] input {
    background: #111827 !important;
    color: #e5e7eb !important;
    border: 1px solid #1f2937 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

div[data-testid="stNumberInput"] input:focus {
    border-color: #00c8ff !important;
}

.stButton button {
    background: linear-gradient(135deg, #0066ff, #00c8ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 16px !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #4b5563;
}

.stTabs [aria-selected="true"] {
    color: #00c8ff !important;
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
            with st.spinner(f"Chargement du modèle..."):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=True)
    model     = joblib.load("models/final_model.joblib")
    scaler    = joblib.load("models/scaler.joblib")
    threshold = joblib.load("models/best_threshold.joblib")
    return model, scaler, threshold

model, scaler, threshold = load_models()

# ── FRAUD EXAMPLE ─────────────────────────────────────────
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

# ── HEADER ────────────────────────────────────────────────
st.markdown(f"""
<div class="header-container">
    <p class="header-title">🛡️ FRAUD<span>GUARD</span></p>
    <p class="header-subtitle">// Credit Card Fraud Detection System — Extra Trees Classifier</p>
    <div style="margin-top: 20px;">
        <span class="metric-pill">PR-AUC <span class="value">0.8808</span></span>
        <span class="metric-pill">Recall <span class="value">84.69%</span></span>
        <span class="metric-pill">Precision <span class="value">93.26%</span></span>
        <span class="metric-pill">F1 <span class="value">0.8877</span></span>
        <span class="metric-pill">Threshold <span class="value">{threshold:.4f}</span></span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD EXAMPLE BUTTON ───────────────────────────────────
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    load_fraud = st.button("⚠️ Charger exemple fraude", use_container_width=True)

# ── INPUT SECTION ─────────────────────────────────────────
st.markdown('<p class="section-title">Transaction Input</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    time_val = st.number_input(
        "⏱ Time (secondes)",
        value=float(FRAUD_EXAMPLE['Time']) if load_fraud else 0.0,
        format="%.2f"
    )
with col2:
    amount_val = st.number_input(
        "💰 Amount (€)",
        value=float(FRAUD_EXAMPLE['Amount']) if load_fraud else 0.0,
        min_value=0.0, format="%.2f"
    )

st.markdown('<p class="section-title">PCA Components — V1 to V28</p>', unsafe_allow_html=True)

# Tabs pour organiser V1-V28
tab1, tab2, tab3, tab4 = st.tabs(["V1 — V7", "V8 — V14", "V15 — V21", "V22 — V28"])

v_values = {}
ranges = [(1,8), (8,15), (15,22), (22,29)]

for tab, (start, end) in zip([tab1, tab2, tab3, tab4], ranges):
    with tab:
        cols = st.columns(7)
        for i, col in zip(range(start, end), cols):
            with col:
                v_values[f'V{i}'] = st.number_input(
                    f"V{i}",
                    value=float(FRAUD_EXAMPLE[f'V{i}']) if load_fraud else 0.0,
                    format="%.4f",
                    key=f"v{i}"
                )

# ── ANALYZE BUTTON ────────────────────────────────────────
st.markdown("")
analyze = st.button("🔍 ANALYSER LA TRANSACTION", use_container_width=True)

# ── PREDICTION ────────────────────────────────────────────
if analyze:
    features = {
        'Time': time_val,
        **{f'V{i}': v_values[f'V{i}'] for i in range(1, 29)},
        'Amount': amount_val
    }
    input_df = pd.DataFrame([features])

    input_scaled = input_df.copy()
    input_scaled[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

    proba = model.predict_proba(input_scaled)[0][1]
    prediction = int(proba >= threshold)

    st.markdown('<p class="section-title">Résultat de l\'analyse</p>', unsafe_allow_html=True)

    col_res, col_prob, col_thresh = st.columns([2, 1, 1])

    with col_res:
        if prediction == 1:
            st.markdown(f"""
            <div class="fraud-result">
                <p class="result-label" style="color: #ef4444;">⚠ FRAUDE</p>
                <p class="result-sub" style="color: #fca5a5;">Transaction suspecte détectée</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-result">
                <p class="result-label" style="color: #10b981;">✓ LÉGITIME</p>
                <p class="result-sub" style="color: #6ee7b7;">Transaction autorisée</p>
            </div>""", unsafe_allow_html=True)

    with col_prob:
        st.markdown(f"""
        <div class="prob-display">
            <p class="prob-value">{proba:.2%}</p>
            <p class="prob-label">Probabilité fraude</p>
        </div>""", unsafe_allow_html=True)

    with col_thresh:
        st.markdown(f"""
        <div class="prob-display">
            <p class="prob-value" style="color: #8b5cf6;">{threshold:.4f}</p>
            <p class="prob-label">Threshold</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.progress(float(proba))

    st.markdown('<p class="section-title">Données saisies</p>', unsafe_allow_html=True)
    st.dataframe(input_df.style.format("{:.4f}"), use_container_width=True)