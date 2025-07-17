import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.exceptions import NotFittedError

# === CONFIG ===
st.set_page_config(page_title="NexusFlow | Neonatal Risk Predictor", layout="wide", page_icon="👶")

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

# === Load Assets ===
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load("model_rf.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, scaler, label_encoder
    except FileNotFoundError as e:
        st.error(f"Required model asset not found: {str(e)}")
        st.stop()

# === Predict Function ===
def predict(model, scaler, label_encoder, data):
    try:
        scaled = scaler.transform(data)
        pred = model.predict(scaled)[0]
        decoded = label_encoder.inverse_transform([pred])[0]
        confidence = np.max(model.predict_proba(scaled)) * 100
        return decoded, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# === MAIN UI ===
st.title("👶 NexusFlow: Neonatal Risk & Birth Weight Predictor")

with st.form("prediction_form"):
    st.subheader("Enter Maternal Data")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Mother's Age (years)", 15, 45, 25)
        pre_preg_bmi = st.number_input("Pre-pregnancy BMI", 10.0, 50.0, 22.0)
        gest_weeks = st.slider("Gestational Age (weeks)", 20, 42, 38)
        systolic = st.number_input("Systolic Blood Pressure (mmHg)", 90, 200, 110)
        diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 50, 140, 70)
        hemoglobin = st.number_input("Hemoglobin Level (g/dl)", 5.0, 18.0, 11.0)

    with col2:
        prenatal_visits = st.slider("Number of Prenatal Visits", 0, 20, 5)
        diabetes = st.radio("Has Diabetes", ["Yes", "No"], horizontal=True)
        hypertension = st.radio("Has Hypertension", ["Yes", "No"], horizontal=True)
        smoking = st.radio("Smoker", ["Yes", "No"], horizontal=True)
        alcohol = st.radio("Alcohol Use", ["Yes", "No"], horizontal=True)
        education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
        income = st.selectbox("Household Income Level", ["Low", "Medium", "High"])
        iron = st.radio("Iron Supplementation", ["Yes", "No"], horizontal=True)

    submitted = st.form_submit_button("🔍 Predict Birth Weight")

if submitted:
    model, scaler, label_encoder = load_model_assets()

    input_data = pd.DataFrame([[
        age,
        pre_preg_bmi,
        gest_weeks,
        systolic,
        diastolic,
        hemoglobin,
        prenatal_visits,
        1 if diabetes == "Yes" else 0,
        1 if hypertension == "Yes" else 0,
        1 if smoking == "Yes" else 0,
        1 if alcohol == "Yes" else 0,
        {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}[education],
        {"Low": 0, "Medium": 1, "High": 2}[income],
        1 if iron == "Yes" else 0
    ]], columns=[
        'age', 'pre_pregnancy_bmi', 'gestational_age_weeks',
        'blood_pressure_systolic', 'blood_pressure_diastolic',
        'hemoglobin_level', 'number_of_prenatal_visits',
        'has_diabetes', 'has_hypertension', 'smoking_status',
        'alcohol_consumption', 'education_level', 'household_income',
        'iron_supplementation'
    ])

    category, confidence = predict(model, scaler, label_encoder, input_data)

    if category:
        color_class = (
            "low-risk" if category == "Normal" else
            "medium-risk" if category == "Low" else
            "high-risk"
        )
        st.markdown(f"""
        <div class="prediction-box {color_class}">
            <h3>Prediction Result</h3>
            <p>Predicted Birth Weight Category: <strong>{category}</strong></p>
            <p>Confidence Score: <strong>{confidence:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Prediction failed.")
