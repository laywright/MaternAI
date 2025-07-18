import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from sklearn.exceptions import NotFittedError

st.set_page_config(page_title="MaternAI 🧠", layout="wide", page_icon="🤱")

# ✨ CSS for gradients, cards, animations
st.markdown("""
<style>
  body {
    background: linear-gradient(135deg, #fff0f5, #ffe4e1);
    font-family: 'Poppins', sans-serif;
  }
  .card {
    background: white;
    padding: 1.5em;
    margin: 1em auto;
    border-radius: 20px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    max-width: 800px;
    animation: fadeIn 1s ease-in-out;
  }
  .stButton>button {
    background-color: #8ab6d6;
    color: white;
    font-weight: bold;
    border-radius: 12px;
    transition: transform 0.2s, background-color 0.3s;
  }
  .stButton>button:hover {
    background-color: #7aa5c6;
    transform: translateY(-2px);
  }
  .progress-bar > div > div {
    background-color: #b2d6c6 !important;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
</style>
""", unsafe_allow_html=True)

# Header + Banner
col1, col2 = st.columns([3,1])
with col1:
    st.title("🤱 MaternAI")
    st.markdown("Predictive neonatal birth weight insights for expectant mothers.")
with col2:
    banner = Image.open("maternal_banner.jpg")  # or use URL
    st.image(banner, width=150)

# Sidebar
with st.sidebar:
    st.header("📋 Steps")
    st.write("1. Fill in details\n2. Click Predict\n3. See results & next steps")
    st.info("All data is private & secure.")

# 🔄 Load needed assets
@st.cache_resource
def load_assets():
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    features = [
        'age','pre_pregnancy_bmi','gestational_age_weeks',
        'blood_pressure_systolic','blood_pressure_diastolic',
        'hemoglobin_level','number_of_prenatal_visits',
        'has_diabetes','has_hypertension',
        'smoking_status','alcohol_consumption',
        'education_level','household_income','iron_supplementation'
    ]
    return model, scaler, encoder, features

def predict(model, scaler, encoder, df, features):
    try:
        df = df[features]
        X = scaler.transform(df)
        pred = model.predict(X)[0]
        prob = np.max(model.predict_proba(X)) * 100
        return encoder.inverse_transform([pred])[0], prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Input Form Card
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Enter Maternal Health Data")

with st.form("form"):
    c1, c2 = st.columns(2)
    # Left
    with c1:
        age = st.slider("Age (years)", 15, 45, 28)
        bmi = st.number_input("Pre-pregnancy BMI", 10.0, 50.0, 23.5)
        ga = st.slider("Gestational age (weeks)", 20, 42, 38)
        sys = st.number_input("Systolic BP", 80, 200, 115)
        dia = st.number_input("Diastolic BP", 50, 130, 70)
    # Right
    with c2:
        hb = st.number_input("Hemoglobin level (g/dl)", 5.0, 18.0, 11.5)
        visits = st.slider("Prenatal visits", 0, 20, 6)
        diabetes = st.radio("Diabetes?", ("Yes", "No"), horizontal=True)
        hypertension = st.radio("Hypertension?", ("Yes", "No"), horizontal=True)
        smoke = st.radio("Smoking?", ("Yes", "No"), horizontal=True)
        alcohol = st.radio("Alcohol?", ("Yes", "No"), horizontal=True)
        education = st.selectbox("Education", ["None", "Primary", "Secondary", "Tertiary"])
        income = st.selectbox("Income", ["Low", "Medium", "High"])
        iron = st.radio("Iron supplements?", ("Yes", "No"), horizontal=True)

    submitted = st.form_submit_button("🔍 Predict Birth Weight")

st.markdown("</div>", unsafe_allow_html=True)

# Prediction & Results
if submitted:
    model, scaler, encoder, features = load_assets()
    df = pd.DataFrame([{
        'age': age, 'pre_pregnancy_bmi': bmi, 'gestational_age_weeks': ga,
        'blood_pressure_systolic': sys, 'blood_pressure_diastolic': dia,
        'hemoglobin_level': hb, 'number_of_prenatal_visits': visits,
        'has_diabetes': int(diabetes=="Yes"), 'has_hypertension': int(hypertension=="Yes"),
        'smoking_status': int(smoke=="Yes"), 'alcohol_consumption': int(alcohol=="Yes"),
        'education_level': ["None","Primary","Secondary","Tertiary"].index(education),
        'household_income': ["Low","Medium","High"].index(income),
        'iron_supplementation': int(iron=="Yes")
    }])

    with st.spinner("Calculating prediction..."):
        category, confidence = predict(model, scaler, encoder, df, features)

    if category:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎯 Result")
        st.success(f"Predicted Category: **{category}**")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(int(confidence))
        st.info("This is a screening tool—consult a healthcare provider for clinical guidance.")
        st.markdown("</div>", unsafe_allow_html=True)

        # CTA Buttons
        cA, cB, cC = st.columns(3)
        with cA: st.button("💾 Save")
        with cB: st.button("📤 Share")
        with cC: st.button("📞 Contact Provider")

# Footer
st.markdown("---")
st.caption("© 2025 MaternAI • Designed for expectant mothers with care 💖")
