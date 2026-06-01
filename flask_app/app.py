from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import gdown
import os
from pathlib import Path

app = Flask(__name__)

# ── DYNAMIC ABSOLUTE PATHS FOR MODELS ─────────────────────
# This finds exactly where app.py lives, and creates a "models" folder right next to it.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Map the exact file paths to their Google Drive IDs
FILES = {
    str(MODEL_DIR / "final_model.joblib"):    "1axYmDQ0i6_qNBKOcV2rz169gGJDYloOF",
    str(MODEL_DIR / "scaler.joblib"):         "1jz20q91NGwtobMoh99tO8XmxczuPyG8F",
    str(MODEL_DIR / "best_threshold.joblib"): "1Ni1zKQwOVjCdNIMPHWFePXe5cP2-EBbN",
}

def load_models():
    for path, file_id in FILES.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
            
    # Load using the exact absolute paths
    model     = joblib.load(str(MODEL_DIR / "final_model.joblib"))
    scaler    = joblib.load(str(MODEL_DIR / "scaler.joblib"))
    threshold = joblib.load(str(MODEL_DIR / "best_threshold.joblib"))
    return model, scaler, threshold

model, scaler, threshold = load_models()

FEATURE_ORDER = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# ── ROUTES ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', threshold=round(float(threshold), 4))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data], columns=FEATURE_ORDER)
        input_scaled = input_df.copy()
        input_scaled[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])
        proba = float(model.predict_proba(input_scaled)[0][1])
        prediction = int(proba >= float(threshold))
        return jsonify({
            'prediction': prediction,
            'probability': round(proba, 4),
            'threshold': round(float(threshold), 4),
            'status': 'FRAUD' if prediction == 1 else 'LEGITIMATE'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)