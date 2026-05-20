import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── CONFIG PAGE ───────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# ── STYLE ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0a0a0f; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
        color: #00ff88 !important;
    }
    
    .fraud-card {
        background: linear-gradient(135deg, #1a0000, #2d0000);
        border: 2px solid #ff3333;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 51, 51, 0.3);
    }
    
    .legit-card {
        background: linear-gradient(135deg, #001a0d, #002d1a);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
    }
    
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    
    .stSlider > div > div { color: #00ff88; }
    
    div[data-testid="stNumberInput"] input {
        background: #111827;
        color: #e5e7eb;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ───────────────────────────────────────────
@st.cache_resource
def load_models():
    model = joblib.load('models/final_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    threshold = joblib.load('models/best_threshold.joblib')
    return model, scaler, threshold

model, scaler, threshold = load_models()

# ── HEADER ────────────────────────────────────────────────
st.markdown("# 🔍 FRAUD DETECTION SYSTEM")
st.markdown("**Détection de transactions frauduleuses — Extra Trees Classifier**")
st.markdown("---")

# ── SIDEBAR INFO ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Modèle")
    st.markdown(f"**Algorithme :** Extra Trees")
    st.markdown(f"**PR-AUC :** 0.8808")
    st.markdown(f"**Recall :** 0.8469")
    st.markdown(f"**Precision :** 0.9326")
    st.markdown(f"**Threshold :** {threshold:.4f}")
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.markdown("V1-V28 sont des composantes PCA anonymisées. Time et Amount sont les seules features interprétables.")

# ── INPUT FORM ────────────────────────────────────────────
st.markdown("## Saisie des Features")
st.markdown("Entrez les caractéristiques de la transaction à analyser.")

col1, col2 = st.columns(2)

with col1:
    time_val = st.number_input("⏱ Time (secondes)", value=0.0, format="%.2f")
    amount_val = st.number_input("💰 Amount (€)", value=0.0, min_value=0.0, format="%.2f")

st.markdown("#### Composantes PCA (V1 — V28)")

# Créer les inputs V1-V28 en grille 4 colonnes
v_values = {}
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i-1) % 4]:
        v_values[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

# ── PREDICTION ────────────────────────────────────────────
st.markdown("---")

if st.button("🔍 Analyser la Transaction", use_container_width=True):
    
    # Construire le vecteur de features
    features = {
        'Time': time_val,
        **{f'V{i}': v_values[f'V{i}'] for i in range(1, 29)},
        'Amount': amount_val
    }
    
    input_df = pd.DataFrame([features])
    
    # Afficher les données saisies
    st.markdown("### 📋 Données Saisies")
    st.dataframe(input_df.style.format("{:.4f}"), use_container_width=True)
    
    # Scaling Time et Amount
    input_scaled = input_df.copy()
    input_scaled[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])
    
    # Prédiction
    proba = model.predict_proba(input_scaled)[0][1]
    prediction = int(proba >= threshold)
    
    # Résultat
    st.markdown("### 🎯 Résultat")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="fraud-card">
                <h2 style="color: #ff3333 !important;">⚠️ FRAUDE</h2>
                <p style="color: #ff9999; font-size: 18px;">Transaction suspecte détectée</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-card">
                <h2 style="color: #00ff88 !important;">✅ LÉGITIME</h2>
                <p style="color: #99ffcc; font-size: 18px;">Transaction autorisée</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #9ca3af; margin: 0;">Probabilité de Fraude</p>
            <h2 style="color: #f59e0b; font-family: 'Space Mono', monospace; margin: 8px 0;">{proba:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #9ca3af; margin: 0;">Threshold Utilisé</p>
            <h2 style="color: #6366f1; font-family: 'Space Mono', monospace; margin: 8px 0;">{threshold:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Barre de probabilité
    st.markdown("### 📈 Niveau de Confiance")
    st.progress(float(proba))
    
    if proba >= threshold:
        st.error(f"⚠️ Probabilité de fraude : **{proba:.2%}** — Au dessus du seuil ({threshold:.4f})")
    else:
        st.success(f"✅ Probabilité de fraude : **{proba:.2%}** — En dessous du seuil ({threshold:.4f})")