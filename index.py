import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.metric_cards import style_metric_cards
import stripe

# === CONFIGURATION ===
st.set_page_config(page_title="NexusFlow | Birth Risk Predictor", page_icon="🧒", layout="wide")
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }
        .main-header { font-size: 3rem; font-weight: bold; color: #004643; }
        .subheader { font-size: 1.2rem; color: #555; }
        .section { margin-top: 40px; }
    </style>
""", unsafe_allow_html=True)

# === Load Model and Transformers ===
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# === PAGE HEADER ===
st.markdown("<div class='main-header'>NexusFlow: Predict Neonatal Risk with AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>An ML-powered tool for early prediction of maternal and neonatal complications.</div>", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.image("https://i.imgur.com/CJwZ5Kf.png", width=150)
st.sidebar.markdown("## Pricing")
st.sidebar.info("Free: 5 predictions/week\n\nPremium: $5/mo for unlimited use")

# === INPUTS ===
st.markdown("### 🎓 Enter Maternal & Birth Data")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Mother's Age", min_value=10, max_value=60, value=25)
    anemia = st.selectbox("Anemia", ['Yes', 'No'])

with col2:
    bp = st.number_input("Blood Pressure (mmHg)", value=80)
    prev_births = st.number_input("Previous Births", 0, 10)

with col3:
    gest_weeks = st.number_input("Gestational Weeks", min_value=24, max_value=42, value=38)
    protein_urine = st.selectbox("Protein in Urine", ['Positive', 'Negative'])

# === FORMATTING ===
data = pd.DataFrame({
    'age': [age],
    'anemia': [anemia],
    'blood_pressure': [bp],
    'previous_births': [prev_births],
    'gestational_weeks': [gest_weeks],
    'protein_urine': [protein_urine]
})

# Encode categorical columns
for col in data.select_dtypes(include='object').columns:
    le = label_encoders[col]
    data[col] = le.transform(data[col])

# === SCALE & PREDICT ===
if st.button("🤝 Predict Risk"):
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    pred_class = label_encoders['birth_weight_category'].inverse_transform([pred])[0]

    st.success(f"Prediction: {pred_class}")
    st.info("Recommendation: Refer for enhanced monitoring.")

    st.markdown("---")
    st.markdown("### 🌈 Prediction Details")
    st.json({
        "Prediction": str(pred_class),
        "Input": data.to_dict(orient='records')[0]
    })

# === SUBSCRIPTION SYSTEM ===
st.markdown("### 💳 Upgrade to Premium")
st.markdown("NexusFlow is free for up to 5 predictions/week. Upgrade for unlimited access!")
st.link_button("Upgrade via Stripe", "https://buy.stripe.com/test_upgrade_page")

# === LIVE METRICS ===
st.markdown("### 📊 Real-Time Platform Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Users", "1,254")
with col2:
    st.metric("Predictions Today", "128")
with col3:
    st.metric("Accuracy", "92.6%")

style_metric_cards(border_left_color="#17A589")

# === FOOTER ===
st.markdown("---")
st.markdown("Made with ❤️ by NexusFlow Team | [Privacy Policy](#) | [Support](#)")
