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
APP_VERSION = "2.3"
MODEL_FILES = {
    "model": "model_rf.pkl",
    "scaler": "scaler.pkl",
    "encoder": "label_encoder.pkl"
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
        
        model = joblib.load(MODEL_FILES["model"])
        scaler = joblib.load(MODEL_FILES["scaler"])
        encoder = joblib.load(MODEL_FILES["encoder"])
        
        # Verify model has expected features
        if not hasattr(model, 'feature_names_in_'):
            raise AttributeError("Model missing feature names information")
            
        return model, scaler, encoder
        
    except Exception as e:
        st.error(f"❌ Error loading model assets: {str(e)}")
        st.error("Please ensure these files exist in your directory:")
        st.error("\n".join(f"- {path}" for path in MODEL_FILES.values()))
        st.stop()

# === Prediction Function ===
def predict_birth_weight(model, scaler, encoder, input_data):
    """Make prediction with proper error handling"""
    try:
        # Ensure correct feature order
        input_data = input_data[model.feature_names_in_]
        
        # Scale and predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        decoded = encoder.inverse_transform([prediction])[0]
        confidence = np.max(model.predict_proba(scaled_data)) * 100
        return decoded, confidence
        
    except KeyError as e:
        st.error(f"Missing required feature: {str(e)}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
    return None, None

# === MAIN APP ===
st.title("👶 NexusFlow: Neonatal Risk Predictor")

# Load models once
model, scaler, encoder = load_model_assets()

# Tabs interface
tab1, tab2 = st.tabs(["Prediction", "Analytics"])

# === PREDICTION TAB ===
with tab1:
    st.subheader("Enter Maternal Data")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Mother's Age (years)", 15, 45, 25)
            systolic = st.number_input("Systolic BP (mmHg)", 80, 200, 110)
            diastolic = st.number_input("Diastolic BP (mmHg)", 40, 120, 70)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 18.0, 11.0, step=0.1)
            parity = st.selectbox("Previous Births", [0, 1, 2, 3, 4, "5+"])
            
        with col2:
            anemia = st.radio("Anemia", ["No", "Yes"], horizontal=True)
            preeclampsia = st.radio("Preeclampsia", ["No", "Yes"], horizontal=True)
            diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=True)
            hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
            smoking = st.radio("Smoking", ["No", "Yes"], horizontal=True)
        
        submit = st.form_submit_button("Predict Birth Weight")

    if submit:
        with st.spinner("Analyzing data..."):
            try:
                # Prepare input data (align with model's expected features)
                input_df = pd.DataFrame({
                    'age': [age],
                    'systolic_bp': [systolic],
                    'diastolic_bp': [diastolic],
                    'hemoglobin': [hemoglobin],
                    'parity': [5 if parity == "5+" else int(parity)],
                    'anemia': [1 if anemia == "Yes" else 0],
                    'preeclampsia': [1 if preeclampsia == "Yes" else 0],
                    'has_diabetes': [1 if diabetes == "Yes" else 0],
                    'has_hypertension': [1 if hypertension == "Yes" else 0],
                    'smoking_status': [1 if smoking == "Yes" else 0]
                })
                
                # Make prediction
                category, confidence = predict_birth_weight(model, scaler, encoder, input_df)
                
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
                            st.success("Normal birth weight (2500-4000g)")
                        elif category == "Low":
                            st.warning("Low birth weight (1500-2500g)")
                        else:
                            st.error("Very low birth weight (<1500g)")
            
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
                              title="Weight Categories")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Feature Correlations")
            numeric_cols = df.select_dtypes(include=np.number).columns
            fig = px.imshow(df[numeric_cols].corr(), 
                           text_auto=True,
                           title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Analytics data not available")
            
    except Exception as e:
        st.error(f"Analytics error: {str(e)}")

# === FOOTER ===
st.markdown(f"""
---
<div style="text-align: center;">
    <p>© {datetime.now().year} NexusFlow · Clinical Decision Support</p>
    <p><small>Version {APP_VERSION}</small></p>
</div>
""", unsafe_allow_html=True)
