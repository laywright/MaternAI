import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.exceptions import NotFittedError

# === CONFIG ===
APP_TITLE = "NexusFlow | Neonatal Risk Predictor"
APP_VERSION = "2.2"

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="👶")

st.title("NexusFlow: Neonatal Risk & Birth Weight Predictor")

@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load("model_rf.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, scaler, label_encoder
    except FileNotFoundError as e:
        st.error(f"Model asset not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading model assets: {e}")
        st.stop()

def predict_birth_weight(model, scaler, label_encoder, data):
    try:
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)[0]
        decoded = label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(model.predict_proba(scaled_data)) * 100
        return decoded, confidence
    except NotFittedError:
        st.error("Model or scaler is not fitted. Please check the training pipeline.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

with st.form("prediction_form"):
    st.subheader("Enter Maternal Data")

    col1, col2 = st.columns(2)

    with col1:
        blood_pressure_systolic = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 110)
        blood_pressure_diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 120, 70)
        number_of_prenatal_visits = st.number_input("Number of Prenatal Visits", 0, 20, 5)
        pre_pregnancy_bmi = st.number_input("Pre-pregnancy BMI", 10.0, 50.0, 22.0, step=0.1)
        mother_age = st.slider("Mother's Age (years)", 15, 45, 25)
        hemoglobin_level = st.number_input("Hemoglobin Level (g/dl)", 5.0, 18.0, 11.0, step=0.1)

    with col2:
        has_diabetes = st.radio("Has Diabetes?", ["No", "Yes"], horizontal=True)
        has_hypertension = st.radio("Has Hypertension?", ["No", "Yes"], horizontal=True)
        household_income = st.selectbox("Household Income Level", ["Low", "Medium", "High"])
        iron_supplementation = st.radio("Iron Supplementation?", ["No", "Yes"], horizontal=True)
        smoking_status = st.radio("Smoker?", ["No", "Yes"], horizontal=True)
        alcohol_consumption = st.radio("Alcohol Use?", ["No", "Yes"], horizontal=True)
        education_level = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
        gestational_age_weeks = st.slider("Gestational Age (weeks)", 20, 42, 38)

    submit = st.form_submit_button("Predict Birth Weight Category")

if submit:
    model, scaler, label_encoder = load_model_assets()

    # Map categorical inputs to numeric as your model expects
    input_df = pd.DataFrame({
        'has_diabetes': [1 if has_diabetes == "Yes" else 0],
        'has_hypertension': [1 if has_hypertension == "Yes" else 0],
        'hemoglobin_level': [hemoglobin_level],
        'household_income': [ {"Low":0, "Medium":1, "High":2}[household_income] ],
        'iron_supplementation': [1 if iron_supplementation == "Yes" else 0],
        'blood_pressure_systolic': [blood_pressure_systolic],
        'blood_pressure_diastolic': [blood_pressure_diastolic],
        'number_of_prenatal_visits': [number_of_prenatal_visits],
        'pre_pregnancy_bmi': [pre_pregnancy_bmi],
        'mother_age': [mother_age],
        'smoking_status': [1 if smoking_status == "Yes" else 0],
        'alcohol_consumption': [1 if alcohol_consumption == "Yes" else 0],
        'education_level': [ {"None":0, "Primary":1, "Secondary":2, "Tertiary":3}[education_level] ],
        'gestational_age_weeks': [gestational_age_weeks]
    })

    category, confidence = predict_birth_weight(model, scaler, label_encoder, input_df)

    risk_class = (
        "low-risk" if category == "Normal"
        else "medium-risk" if category == "Low"
        else "high-risk"
    )

    st.markdown(f"""
        <div style='padding: 1em; border-radius: 0.5em; margin-top: 1em; 
                    background-color: {"#d4edda" if risk_class=="low-risk" else "#fff3cd" if risk_class=="medium-risk" else "#f8d7da"};
                    color: {"#155724" if risk_class=="low-risk" else "#856404" if risk_class=="medium-risk" else "#721c24"}'>
            <h3>Prediction Result</h3>
            <p>Predicted Birth Weight Category: <strong>{category}</strong></p>
            <p>Confidence Score: <strong>{confidence:.1f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("What does this prediction mean?"):
        if category == "Normal":
            st.success("Normal birth weight (2500-4000g). Likely healthy outcome.")
        elif category == "Low":
            st.warning("Low birth weight (1500-2500g). May need special care.")
        else:
            st.error("Very low birth weight (<1500g). High risk of complications.")
