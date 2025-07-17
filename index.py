import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model, scaler, and label encoder ===
@st.cache_resource
def load_assets():
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

# === Prediction function ===
def predict(model, scaler, label_encoder, input_df):
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]
    confidence = np.max(model.predict_proba(scaled)) * 100
    return pred_label, confidence

# === Streamlit UI ===
st.title("NexusFlow | Neonatal Risk & Birth Weight Predictor")

with st.form("prediction_form"):
    st.subheader("Enter Maternal Data")

    col1, col2 = st.columns(2)

    with col1:
        blood_pressure_systolic = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 110)
        blood_pressure_diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 120, 70)
        number_of_prenatal_visits = st.number_input("Number of Prenatal Visits", 0, 20, 5)
        pre_pregnancy_bmi = st.number_input("Pre-pregnancy BMI", 10.0, 50.0, 22.0, step=0.1)
        age = st.slider("Mother's Age (years)", 15, 45, 25)
        hemoglobin = st.number_input("Hemoglobin Level (g/dl)", 5.0, 18.0, 11.0, step=0.1)
        parity = st.selectbox("Number of Previous Births", [0, 1, 2, 3, 4, "5+"])

    with col2:
        anemia = st.radio("Anemia Present", ["Yes", "No"], horizontal=True)
        preeclampsia = st.radio("History of Preeclampsia", ["Yes", "No"], horizontal=True)
        smoking_status = st.radio("Smoker", ["Yes", "No"], horizontal=True)
        alcohol_consumption = st.radio("Alcohol Use", ["Yes", "No"], horizontal=True)
        education_level = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
        gestational_age_weeks = st.slider("Gestational Age (weeks)", 20, 42, 38)

    submit = st.form_submit_button("Predict Birth Weight Category")

if submit:
    model, scaler, label_encoder = load_assets()

    # Map categorical inputs to numerical values consistent with training
    parity_val = 5 if parity == "5+" else int(parity)
    anemia_val = 1 if anemia == "Yes" else 0
    preeclampsia_val = 1 if preeclampsia == "Yes" else 0
    smoking_val = 1 if smoking_status == "Yes" else 0
    alcohol_val = 1 if alcohol_consumption == "Yes" else 0
    education_map = {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}
    education_val = education_map[education_level]

    input_df = pd.DataFrame({
        'blood_pressure_systolic': [blood_pressure_systolic],
        'blood_pressure_diastolic': [blood_pressure_diastolic],
        'number_of_prenatal_visits': [number_of_prenatal_visits],
        'pre_pregnancy_bmi': [pre_pregnancy_bmi],
        'age': [age],
        'hemoglobin': [hemoglobin],
        'parity': [parity_val],
        'anemia': [anemia_val],
        'preeclampsia': [preeclampsia_val],
        'smoking_status': [smoking_val],
        'alcohol_consumption': [alcohol_val],
        'education_level': [education_val],
        'gestational_age_weeks': [gestational_age_weeks]
    })

    try:
        category, confidence = predict(model, scaler, label_encoder, input_df)
        risk_class = "low-risk" if category == "Normal" else "medium-risk" if category == "Low" else "high-risk"

        st.markdown(f"""
        <div style="padding: 1.5em; border-radius: 0.5em; margin: 1em 0;
                    background-color: {'#d4edda' if risk_class=='low-risk' else '#fff3cd' if risk_class=='medium-risk' else '#f8d7da'};
                    color: {'#155724' if risk_class=='low-risk' else '#856404' if risk_class=='medium-risk' else '#721c24'};">
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

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
