# app.py - Fixed Version

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# === CONFIG ===
MODEL_FILES = {
    "model": "model_rf.pkl",
    "scaler": "scaler.pkl",
    "encoder": "label_encoder.pkl"
}

# === UI SETUP ===
st.set_page_config(page_title="Neonatal Risk Predictor", layout="centered")
st.title("👶 Birth Weight Predictor")

# === MODEL LOADING ===
@st.cache_resource
def load_models():
    try:
        return (
            joblib.load(MODEL_FILES["model"]),
            joblib.load(MODEL_FILES["scaler"]),
            joblib.load(MODEL_FILES["encoder"])
        )
    except FileNotFoundError as e:
        st.error(f"Missing model file: {str(e)}")
        st.stop()

model, scaler, encoder = load_models()

# === PREDICTION FORM ===
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 18.0, 12.0, step=0.1)
        
    with col2:
        parity = st.selectbox("Previous Births", [0, 1, 2, 3, 4, "5+"])
        anemia = st.radio("Anemia", ["No", "Yes"], horizontal=True)
        preeclampsia = st.radio("Preeclampsia", ["No", "Yes"], horizontal=True)
    
    submit = st.form_submit_button("Predict")

# === PREDICTION LOGIC ===
if submit:
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'systolic_bp': [systolic_bp],
            'diastolic_bp': [diastolic_bp],
            'hemoglobin': [hemoglobin],
            'parity': [5 if parity == "5+" else int(parity)],
            'anemia': [1 if anemia == "Yes" else 0],
            'preeclampsia': [1 if preeclampsia == "Yes" else 0]
        })[model.feature_names_in_]  # Ensure correct order
        
        # Make prediction
        scaled_data = scaler.transform(input_data)
        pred = model.predict(scaled_data)[0]
        pred_label = encoder.inverse_transform([pred])[0]
        confidence = np.max(model.predict_proba(scaled_data)) * 100
        
        # Show results
        risk_class = "low-risk" if pred_label == "Normal" else "medium-risk" if pred_label == "Low" else "high-risk"
        st.markdown(f"""
        <div style="padding:1.5em; border-radius:0.5em; margin:1em 0; background-color:{{
            'low-risk':'#d4edda', 
            'medium-risk':'#fff3cd', 
            'high-risk':'#f8d7da'
        }[risk_class]}}">
            <h3>Result: {pred_label}</h3>
            <p>Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# === FOOTER ===
st.markdown(f"""
---
<small>Version 1.0 · {datetime.now().year}</small>
""", unsafe_allow_html=True)
