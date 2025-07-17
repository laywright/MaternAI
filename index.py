# app.py (Simplified NexusFlow Neonatal Predictor)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px

# === 🎨 UI CONFIG ===
st.set_page_config(
    page_title="NexusFlow | Neonatal Risk Predictor", 
    layout="wide",
    page_icon="👶"
)

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

# === 📄 TITLE ===
st.title("👶 NexusFlow: Neonatal Risk & Birth Weight Predictor")

# === 📥 INPUT SECTION ===
st.subheader("📋 Enter Maternal Data")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Mother's Age (years)", 15, 45, 25, help="Age range 15-45 years")
        systolic = st.number_input("Systolic BP (mmHg)", 80, 200, 110, 
                                 help="Normal range: 90-120 mmHg")
        diastolic = st.number_input("Diastolic BP (mmHg)", 40, 120, 70,
                                  help="Normal range: 60-80 mmHg")
        
    with col2:
        hemoglobin = st.number_input("Hemoglobin Level (g/dl)", 5.0, 18.0, 11.0, step=0.1,
                                    help="Normal range: 12-16 g/dl")
        parity = st.selectbox("Number of Previous Births", [0, 1, 2, 3, 4, "5+"],
                            help="Including stillbirths and miscarriages")
        anemia = st.radio("Anemia Present", ["Yes", "No"], horizontal=True)
        preeclampsia = st.radio("History of Preeclampsia", ["Yes", "No"], horizontal=True)
    
    submit = st.form_submit_button("Predict Birth Weight Category")

# === 🧠 MODEL & PREDICTION ===
def load_model_assets():
    """Load ML model and preprocessing assets"""
    try:
        model = joblib.load("model_rf.pkl")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, scaler, le
    except FileNotFoundError as e:
        st.error(f"Model file not found: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

if submit:
    with st.spinner("Analyzing data and making prediction..."):
        try:
            # Load model
            model, scaler, le = load_model_assets()
            
            # Process parity input
            parity_val = 5 if parity == "5+" else int(parity)
            
            # Prepare input data
            input_data = pd.DataFrame({
                'age': [age],
                'systolic_bp': [systolic],
                'diastolic_bp': [diastolic],
                'hemoglobin': [hemoglobin],
                'parity': [parity_val],
                'anemia': [1 if anemia == "Yes" else 0],
                'preeclampsia': [1 if preeclampsia == "Yes" else 0]
            })

            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            decoded = le.inverse_transform([prediction])[0]
            
            # Display results with appropriate styling
            risk_level = "low-risk" if decoded == "Normal" else "medium-risk" if decoded == "Low" else "high-risk"
            st.markdown(f"""
            <div class="prediction-box {risk_level}">
                <h3>Prediction Result</h3>
                <p>Predicted Birth Weight Category: <strong>{decoded}</strong></p>
                <p>Confidence: <strong>{np.max(model.predict_proba(input_scaled))*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Explanation
            with st.expander("What does this prediction mean?"):
                if decoded == "Normal":
                    st.info("Normal birth weight (2500-4000g). The baby is likely to be born with healthy weight.")
                elif decoded == "Low":
                    st.warning("Low birth weight (1500-2500g). The baby may need special care after birth.")
                else:
                    st.error("Very low birth weight (<1500g). High risk of complications requiring NICU care.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# === 📌 FOOTER ===
st.markdown("""
---
<div style="text-align: center;">
    <p>© 2025 NexusFlow · Built for SDG 3.1 · <a href="mailto:info@nexusflow.ai">Contact Us</a></p>
    <p><small>Version 2.1 · Last updated: {}</small></p>
</div>
""".format(pd.Timestamp.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
