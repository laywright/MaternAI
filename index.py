# app.py - Final Corrected Version

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.exceptions import NotFittedError
import os
from datetime import datetime

# === CONFIG ===
APP_TITLE = "NexusFlow | Neonatal Risk Predictor"
APP_VERSION = "2.4"
MODEL_FILES = {
    "model": "model_rf.pkl",
    "scaler": "scaler.pkl",
    "encoder": "label_encoder.pkl",
    "columns": "columns.pkl"
}

# === 🎨 UI CONFIG ===
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="👶")

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    h1 {color: #0072e5;}
    .stButton>button {background-color: #0072e5; color: white; border-radius: 0.5em;}
    .prediction-box {padding: 1.5em; border-radius: 0.5em; margin: 1em 0;}
    .low-risk {background-color: #d4edda; color: #155724;}
    .medium-risk {background-color: #fff3cd; color: #856404;}
    .high-risk {background-color: #f8d7da; color: #721c24;}
</style>
""", unsafe_allow_html=True)

# === Model Loading ===
@st.cache_resource
def load_model_assets():
    """Load all required model files with error handling"""
    try:
        # Verify files exist
        missing_files = [name for name, path in MODEL_FILES.items() if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")
        
        return (
            joblib.load(MODEL_FILES["model"]),
            joblib.load(MODEL_FILES["scaler"]),
            joblib.load(MODEL_FILES["encoder"]),
            joblib.load(MODEL_FILES["columns"])
        )
    except Exception as e:
        st.error(f"❌ Error loading model assets: {str(e)}")
        st.error("Please ensure these files exist in your directory:")
        st.error("\n".join(f"- {path}" for path in MODEL_FILES.values()))
        st.stop()

# === Prediction Function ===
def predict_birth_weight(model, scaler, encoder, data, columns):
    """Make prediction with proper error handling"""
    try:
        # Ensure correct feature order and presence
        missing_cols = set(columns) - set(data.columns)
        extra_cols = set(data.columns) - set(columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        if extra_cols:
            st.warning(f"Extra columns provided: {extra_cols}")
        
        data = data[columns]  # Enforce column order
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)[0]
        decoded = encoder.inverse_transform([prediction])[0]
        confidence = np.max(model.predict_proba(scaled_data)) * 100
        return decoded, confidence
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

# === MAIN APP ===
st.title("👶 NexusFlow: Neonatal Risk Predictor")

# Load models
model, scaler, encoder, model_columns = load_model_assets()

# Tabs interface
tab1, tab2 = st.tabs(["Prediction", "Analytics"])

# === PREDICTION TAB ===
with tab1:
    st.subheader("Enter Maternal Data")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Mother's Age (years)", 15, 45, 25)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 18.0, 11.0, step=0.1)
            gestational_age = st.slider("Gestational Age (weeks)", 20, 42, 38)
            systolic = st.number_input("Systolic BP (mmHg)", 80, 200, 110)
            diastolic = st.number_input("Diastolic BP (mmHg)", 40, 120, 70)
            
        with col2:
            parity = st.selectbox("Previous Births", [0, 1, 2, 3, 4, "5+"])
            anemia = st.radio("Anemia", ["No", "Yes"], horizontal=True)
            preeclampsia = st.radio("Preeclampsia", ["No", "Yes"], horizontal=True)
            diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=True)
            hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        
        submit = st.form_submit_button("Predict Birth Weight")

    if submit:
        with st.spinner("Analyzing data..."):
            try:
                # Prepare input data
                input_df = pd.DataFrame({
                    'age': [age],
                    'hemoglobin_level': [hemoglobin],
                    'gestational_age_weeks': [gestational_age],
                    'systolic_bp': [systolic],
                    'diastolic_bp': [diastolic],
                    'parity': [5 if parity == "5+" else int(parity)],
                    'anemia': [1 if anemia == "Yes" else 0],
                    'preeclampsia': [1 if preeclampsia == "Yes" else 0],
                    'has_diabetes': [1 if diabetes == "Yes" else 0],
                    'has_hypertension': [1 if hypertension == "Yes" else 0]
                })
                
                # Make prediction
                category, confidence = predict_birth_weight(
                    model, scaler, encoder, input_df, model_columns
                )
                
                if category:
                    risk_class = "low-risk" if category == "Normal" else "medium-risk" if category == "Low" else "high-risk"
                    
                    st.markdown(f"""
                    <div class="prediction-box {risk_class}">
                        <h3>Prediction Result</h3>
                        <p>Birth Weight: <strong>{category}</strong></p>
                        <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Clinical Interpretation"):
                        if category == "Normal":
                            st.success("Normal range (2500-4000g). Standard care recommended.")
                        elif category == "Low":
                            st.warning("Low birth weight (1500-2500g). Monitor closely.")
                        else:
                            st.error("Very low birth weight (<1500g). High-risk case.")
                else:
                    st.error("Prediction failed. Please check your inputs.")
            
            except Exception as e:
                st.error(f"Application error: {str(e)}")

# === ANALYTICS TAB ===
with tab2:
    st.subheader("Data Insights")
    
    try:
        if os.path.exists("analytics_sample.csv"):
            df = pd.read_csv("analytics_sample.csv")
            
            st.markdown("### Birth Weight Distribution")
            fig = px.histogram(df, x="birth_weight_category", 
                              color="birth_weight_category",
                              title="Category Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Feature Correlations")
            numeric_cols = df.select_dtypes(include=np.number).columns
            fig = px.imshow(df[numeric_cols].corr(), 
                           text_auto=True,
                           title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Analytics data not available. Sample data would appear here.")
            
    except Exception as e:
        st.error(f"Could not generate analytics: {str(e)}")

# === FOOTER ===
st.markdown(f"""
---
<div style="text-align: center;">
    <p>© {datetime.now().year} NexusFlow · Clinical Decision Support System</p>
    <p><small>Version {APP_VERSION} · For professional use only</small></p>
</div>
""", unsafe_allow_html=True)
