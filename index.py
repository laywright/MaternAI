import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.exceptions import NotFittedError
from streamlit_lottie import st_lottie

# Page config
st.set_page_config(page_title="MaternAI - Neonatal Risk Predictor", layout="wide", page_icon="👶")

# === Custom Styling ===
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #fff0f5, #ffe4e1);
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2em;
    }
    .stButton > button {
        background-color: #f6b8b8;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1.2em;
    }
    .stProgress > div > div {
        background-color: #8ab6d6;
    }
    .card {
        background-color: white;
        padding: 1.5em;
        margin-bottom: 1.2em;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤱 MaternAI: Neonatal Birth Weight Predictor")
st.markdown("Empowering maternal wellness through data-driven prediction.")

# === Load Lottie Animation ===
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_cgqfuzsr.json")
st_lottie(lottie_animation, height=200)

# === Load Model Assets ===
@st.cache_resource
def load_assets():
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_order = [
        'age', 'pre_pregnancy_bmi', 'gestational_age_weeks', 'blood_pressure_systolic',
        'blood_pressure_diastolic', 'hemoglobin_level', 'number_of_prenatal_visits',
        'has_diabetes', 'has_hypertension', 'smoking_status', 'alcohol_consumption',
        'education_level', 'household_income', 'iron_supplementation'
    ]
    return model, scaler, label_encoder, feature_order

# === Prediction Function ===
def predict(model, scaler, label_encoder, input_df, feature_order):
    try:
        input_df = input_df[feature_order]
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

# === Educational Tooltips ===
tooltips = {
    "bmi": "Body Mass Index (BMI) before pregnancy.",
    "gestation": "Gestational age in weeks (typically 37–42 weeks).",
    "bp_sys": "Systolic pressure (top number). Normal: 90–120 mmHg.",
    "bp_dia": "Diastolic pressure (bottom number). Normal: 60–80 mmHg.",
    "hb": "Hemoglobin level. Normal: 11–16 g/dl during pregnancy.",
    "visits": "Recommended 8+ visits during pregnancy."
}

# === Sidebar Navigation ===
with st.sidebar:
    st.header("📋 Navigation")
    st.markdown("""
        - 📅 Fill maternal data
        - 🔍 Predict outcomes
        - 📊 View insights
    """)
    st.markdown("Made with ❤️ for maternal health.")
    st.markdown("[WHO Maternal Health Info](https://www.who.int/health-topics/maternal-health)")

# === Input Form ===
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📥 Enter Maternal Health Information")
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Mother's Age (years)", 15, 45, 25)
        pre_pregnancy_bmi = st.number_input("Pre-pregnancy BMI", 10.0, 50.0, 22.0, help=tooltips["bmi"])
        gestational_age = st.slider("Gestational Age (weeks)", 20, 42, 38, help=tooltips["gestation"])
        systolic = st.number_input("Systolic BP (mmHg)", 80, 200, 110, help=tooltips["bp_sys"])
        diastolic = st.number_input("Diastolic BP (mmHg)", 50, 130, 70, help=tooltips["bp_dia"])
        hemoglobin = st.number_input("Hemoglobin Level (g/dl)", 5.0, 18.0, 11.0, help=tooltips["hb"])
        prenatal_visits = st.slider("Number of Prenatal Visits", 0, 20, 5, help=tooltips["visits"])

    with col2:
        diabetes = st.radio("Has Diabetes?", ["Yes", "No"], horizontal=True)
        hypertension = st.radio("Has Hypertension?", ["Yes", "No"], horizontal=True)
        smoking = st.radio("Smoker?", ["Yes", "No"], horizontal=True)
        alcohol = st.radio("Alcohol Consumption?", ["Yes", "No"], horizontal=True)
        education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
        income = st.selectbox("Household Income", ["Low", "Medium", "High"])
        iron = st.radio("Iron Supplementation?", ["Yes", "No"], horizontal=True)

    submitted = st.form_submit_button("🚀 Predict Birth Weight Category")
st.markdown("</div>", unsafe_allow_html=True)

# === Handle Submission ===
if submitted:
    with st.spinner("Analyzing maternal health data..."):
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

        category, confidence = predict(model, scaler, label_encoder, input_data, feature_order)

    if category:
        st.success(f"🎯 Predicted Birth Weight Category: **{category}**")
        st.metric("Confidence Level", f"{confidence:.2f}%")
        st.progress(int(confidence))
        st.markdown(f"""
        <div style='background-color: #fff7f3; padding: 1em; border-radius: 10px;'>
            <b>Note:</b> This prediction is a screening tool. For a comprehensive assessment, please consult a qualified medical professional.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Prediction failed. Please review the input data.")

# === Call to Action ===
st.markdown("---")
st.subheader("📌 Next Steps")
colA, colB, colC = st.columns(3)
with colA:
    st.button("📀 Save Result")
with colB:
    st.button("📤 Share with Doctor")
with colC:
    st.button("📞 Seek Guidance")

st.caption("Made with 🧬 and ❤️ | © 2025 MaternAI")
