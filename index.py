import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.exceptions import NotFittedError

# App config
st.set_page_config(page_title="Neonatal Risk Predictor", layout="centered")

st.title("👶 Neonatal Risk & Birth Weight Predictor")

# Load model assets
@st.cache_resource
def load_assets():
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    feature_order = [
        'age',
        'pre_pregnancy_bmi',
        'gestational_age_weeks',
        'blood_pressure_systolic',
        'blood_pressure_diastolic',
        'hemoglobin_level',
        'number_of_prenatal_visits',
        'has_diabetes',
        'has_hypertension',
        'smoking_status',
        'alcohol_consumption',
        'education_level',
        'household_income',
        'iron_supplementation'
    ]
    return model, scaler, label_encoder, feature_order

# Prediction function
def predict(model, scaler, label_encoder, input_df, feature_order):
    try:
        input_df = input_df[feature_order]  # Ensure column order
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        prob = np.max(model.predict_proba(scaled)) * 100
        decoded = label_encoder.inverse_transform([pred])[0]
        return decoded, prob
    except NotFittedError:
        st.error("Model not fitted.")
        return None, None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# === User Input Form ===
st.subheader("Enter Maternal Information")
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Mother's Age (years)", 15, 45, 25)
        pre_pregnancy_bmi = st.number_input("Pre-pregnancy BMI", 10.0, 50.0, 22.0)
        gestational_age = st.slider("Gestational Age (weeks)", 20, 42, 38)
        systolic = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 110)
        diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 50, 130, 70)
        hemoglobin = st.number_input("Hemoglobin Level (g/dl)", 5.0, 18.0, 11.0)
        prenatal_visits = st.slider("Number of Prenatal Visits", 0, 20, 5)

    with col2:
        diabetes = st.radio("Has Diabetes?", ["Yes", "No"], horizontal=True)
        hypertension = st.radio("Has Hypertension?", ["Yes", "No"], horizontal=True)
        smoking = st.radio("Smoker?", ["Yes", "No"], horizontal=True)
        alcohol = st.radio("Alcohol Consumption?", ["Yes", "No"], horizontal=True)
        education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
        income = st.selectbox("Household Income", ["Low", "Medium", "High"])
        iron = st.radio("Iron Supplementation?", ["Yes", "No"], horizontal=True)

    submitted = st.form_submit_button("Predict Birth Weight Category")

# === Handle Submission ===
if submitted:
    model, scaler, label_encoder, feature_order = load_assets()

    input_data = pd.DataFrame([{
        'age': age,
        'pre_pregnancy_bmi': pre_pregnancy_bmi,
        'gestational_age_weeks': gestational_age,
        'blood_pressure_systolic': systolic,
        'blood_pressure_diastolic': diastolic,
        'hemoglobin_level': hemoglobin,
        'number_of_prenatal_visits': prenatal_visits,
        'has_diabetes': 1 if diabetes == "Yes" else 0,
        'has_hypertension': 1 if hypertension == "Yes" else 0,
        'smoking_status': 1 if smoking == "Yes" else 0,
        'alcohol_consumption': 1 if alcohol == "Yes" else 0,
        'education_level': {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}[education],
        'household_income': {"Low": 0, "Medium": 1, "High": 2}[income],
        'iron_supplementation': 1 if iron == "Yes" else 0
    }])

    with st.spinner("Analyzing..."):
        category, confidence = predict(model, scaler, label_encoder, input_data, feature_order)

        if category:
            st.success(f"Prediction: **{category}**")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.error("Prediction failed.")
