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
        
        return (
            joblib.load(MODEL_FILES["model"]),
            joblib.load(MODEL_FILES["scaler"]),
            joblib.load(MODEL_FILES["encoder"])
        )
    except Exception as e:
        st.error(f"❌ Error loading model assets: {str(e)}")
        st.error("Please ensure these files exist in your directory:")
        st.error("\n".join(f"- {path}" for path in MODEL_FILES.values()))
        st.stop()

# === Prediction Function ===
def predict_birth_weight(model, scaler, label_encoder, input_data):
    """Make prediction with proper error handling"""
    try:
        # Validate input shape
        if input_data.shape[1] != len(model.feature_names_in_):
            raise ValueError(f"Expected {len(model.feature_names_in_)} features, got {input_data.shape[1]}")
        
        # Scale and predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        decoded = label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(model.predict_proba(scaled_data)) * 100
        return decoded, confidence
        
    except NotFittedError:
        st.error("Model not properly trained. Check your training pipeline.")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
    return None, None

# === MAIN APP ===
st.title("👶 NexusFlow: Neonatal Risk Predictor")

# Initialize session state
if 'model_loaded' not in st.session_state:
    try:
        st.session_state.model, st.session_state.scaler, st.session_state.encoder = load_model_assets()
        st.session_state.model_loaded = True
    except:
        st.session_state.model_loaded = False

# Tabs interface
tab1, tab2 = st.tabs(["Prediction", "Analytics"])

# === PREDICTION TAB ===
with tab1:
    st.subheader("Enter Maternal Health Data")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Mother's Age (years)", 15, 45, 25)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 18.0, 11.0, step=0.1)
            gestational_age = st.slider("Gestational Age (weeks)", 20, 42, 38)
            income = st.selectbox("Household Income", ["Low", "Medium", "High"])
            iron = st.radio("Iron Supplements", ["Yes", "No"], horizontal=True)
            
        with col2:
            diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=True)
            hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
            smoking = st.radio("Smoking Status", ["No", "Yes"], horizontal=True)
            alcohol = st.radio("Alcohol Use", ["No", "Yes"], horizontal=True)
            education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
        
        submitted = st.form_submit_button("Predict Birth Weight")

    if submitted and st.session_state.model_loaded:
        try:
            # Prepare input data
            input_df = pd.DataFrame({
                'age': [age],
                'has_diabetes': [1 if diabetes == "Yes" else 0],
                'has_hypertension': [1 if hypertension == "Yes" else 0],
                'hemoglobin_level': [hemoglobin],
                'household_income': ["Low", "Medium", "High"].index(income),
                'iron_supplementation': [1 if iron == "Yes" else 0],
                'smoking_status': [1 if smoking == "Yes" else 0],
                'alcohol_consumption': [1 if alcohol == "Yes" else 0],
                'education_level': ["None", "Primary", "Secondary", "Tertiary"].index(education),
                'gestational_age_weeks': [gestational_age]
            })
            
            # Ensure correct feature order
            input_df = input_df[st.session_state.model.feature_names_in_]
            
            # Make prediction
            with st.spinner("Analyzing..."):
                category, confidence = predict_birth_weight(
                    st.session_state.model,
                    st.session_state.scaler,
                    st.session_state.encoder,
                    input_df
                )
                
                if category:
                    risk_class = "low-risk" if category == "Normal" else "medium-risk" if category == "Low" else "high-risk"
                    
                    st.markdown(f"""
                    <div class="prediction-box {risk_class}">
                        <h3>Prediction Result</h3>
                        <p><strong>Birth Weight Category:</strong> {category}</p>
                        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Clinical interpretation
                    with st.expander("Clinical Notes"):
                        if category == "Normal":
                            st.info("Normal range (2500-4000g). Standard care recommended.")
                        elif category == "Low":
                            st.warning("Low birth weight (1500-2500g). Monitor closely.")
                        else:
                            st.error("Very low birth weight (<1500g). High-risk case.")
        
        except Exception as e:
            st.error(f"Application error: {str(e)}")

# === ANALYTICS TAB ===
with tab2:
    st.subheader("Data Insights")
    
    try:
        if os.path.exists("analytics_sample.csv"):
            df = pd.read_csv("analytics_sample.csv")
            
            st.markdown("### Birth Weight Distribution")
            fig1 = px.histogram(df, x="birth_weight_category", 
                               color="birth_weight_category",
                               title="Category Distribution")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("### Feature Correlations")
            numeric_cols = df.select_dtypes(include=np.number).columns
            fig2 = px.imshow(df[numeric_cols].corr(), 
                            text_auto=True,
                            title="Correlation Matrix")
            st.plotly_chart(fig2, use_container_width=True)
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
