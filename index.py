# app.py - MaternAI Neonatal Risk Predictor (Simplified)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import os

# === 🛠️ CONFIGURATION ===
MODEL_PATH = "model_rf.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoders.pkl"
APP_VERSION = "3.1"

# === 🎨 UI CONFIG ===
def setup_ui():
    st.set_page_config(
        page_title="MaternAI | Neonatal Risk Predictor",
        layout="centered",
        page_icon="👶"
    )
    
    st.markdown("""
    <style>
        .prediction-box {padding: 1.5em; border-radius: 0.5em; margin: 1em 0;}
        .low-risk {background-color: #d4edda; color: #155724;}
        .medium-risk {background-color: #fff3cd; color: #856404;}
        .high-risk {background-color: #f8d7da; color: #721c24;}
    </style>
    """, unsafe_allow_html=True)

# === 🧠 MODEL HANDLING ===
def load_assets():
    try:
        return (
            joblib.load(MODEL_PATH),
            joblib.load(SCALER_PATH),
            joblib.load(LABEL_ENCODER_PATH)
        )
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.stop()

# === 📋 MAIN APP ===
def main():
    setup_ui()
    
    # Load assets once
    model, scaler, label_encoders = load_assets()
    
    # App header
    st.title("👶 MaternAI Birth Weight Predictor")
    st.write(f"*Version {APP_VERSION}*")
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Maternal Health Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", 15, 45, 25)
            systolic = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 18.0, 12.0, step=0.1)
            
        with col2:
            diastolic = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)
            parity = st.selectbox("Previous Births", [0, 1, 2, 3, 4, "5+"])
            diabetes = st.checkbox("Diabetes")
            hypertension = st.checkbox("Hypertension")
        
        submitted = st.form_submit_button("Predict Birth Weight")

    # Prediction logic
    if submitted:
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'age': [age],
                'blood_pressure_systolic': [systolic],
                'blood_pressure_diastolic': [diastolic],
                'hemoglobin_level': [hemoglobin],
                'parity': [5 if parity == "5+" else int(parity)],
                'has_diabetes': [int(diabetes)],
                'has_hypertension': [int(hypertension)],
                'household_income': [3]  # Default middle income
            })
            
            # Scale features
            scaled_data = scaler.transform(input_data)
            
            # Make prediction
            pred = model.predict(scaled_data)[0]
            proba = model.predict_proba(scaled_data)[0]
            confidence = np.max(proba) * 100
            
            # Decode prediction
            pred_label = label_encoders['birth_weight_category'].inverse_transform([pred])[0]
            
            # Display results
            risk_level = "low-risk" if pred_label == "Normal" else "medium-risk" if pred_label == "Low" else "high-risk"
            st.markdown(f"""
            <div class="prediction-box {risk_level}">
                <h3>Prediction Result</h3>
                <p><strong>Birth Weight Category:</strong> {pred_label}</p>
                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Clinical notes
            with st.expander("Clinical Interpretation"):
                if pred_label == "Normal":
                    st.info("Expected normal birth weight (2500-4000g)")
                elif pred_label == "Low":
                    st.warning("Potential low birth weight risk (1500-2500g)")
                else:
                    st.error("High risk of very low birth weight (<1500g)")
                    
        except Exception as e:
            st.error(f"⚠️ Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
