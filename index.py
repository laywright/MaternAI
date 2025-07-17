# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.exceptions import NotFittedError

# === CONFIG ===
APP_TITLE = "NexusFlow | Neonatal Risk Predictor"
APP_VERSION = "2.3"

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

# === Load Assets ===
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load("model_rf.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        column_order = joblib.load("columns.pkl")
        return model, scaler, label_encoder, column_order
    except FileNotFoundError as e:
        st.error(f"Required model asset not found: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading model assets: {str(e)}")
        st.stop()

# === Predict Function ===
def predict_birth_weight(model, scaler, label_encoder, data, column_order):
    try:
        data = data[column_order]  # Ensure correct column order
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)[0]
        decoded = label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(model.predict_proba(scaled_data)) * 100
        return decoded, confidence
    except NotFittedError:
        st.error("Model or scaler is not fitted.")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# === MAIN UI ===
st.title("NexusFlow: Neonatal Risk & Birth Weight Predictor")

# Tabs for Prediction & Analytics
tabs = st.tabs(["Prediction", "Analytics"])

# === Prediction Tab ===
with tabs[0]:
    st.subheader("Enter Maternal Data")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Mother's Age (years)", 15, 45, 25)
            hemoglobin_level = st.number_input("Hemoglobin Level (g/dl)", 5.0, 18.0, 11.0, step=0.1)
            gestational_age = st.slider("Gestational Age (weeks)", 20, 42, 38)
            household_income = st.selectbox("Household Income Level", ["Low", "Medium", "High"])
            iron_supp = st.radio("Iron Supplementation", ["Yes", "No"], horizontal=True)
        with col2:
            diabetes = st.radio("Has Diabetes", ["Yes", "No"], horizontal=True)
            hypertension = st.radio("Has Hypertension", ["Yes", "No"], horizontal=True)
            smoking_status = st.radio("Smoker", ["Yes", "No"], horizontal=True)
            alcohol = st.radio("Alcohol Use", ["Yes", "No"], horizontal=True)
            education_level = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])

        submit = st.form_submit_button("Predict Birth Weight Category")

    if submit:
        model, scaler, label_encoder, column_order = load_model_assets()
        with st.spinner("Analyzing data and making prediction..."):
            try:
                input_df = pd.DataFrame({
                    'age': [age],
                    'has_diabetes': [1 if diabetes == "Yes" else 0],
                    'has_hypertension': [1 if hypertension == "Yes" else 0],
                    'hemoglobin_level': [hemoglobin_level],
                    'household_income': [ {"Low": 0, "Medium": 1, "High": 2}[household_income] ],
                    'iron_supplementation': [1 if iron_supp == "Yes" else 0],
                    'smoking_status': [1 if smoking_status == "Yes" else 0],
                    'alcohol_consumption': [1 if alcohol == "Yes" else 0],
                    'education_level': [ {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}[education_level] ],
                    'gestational_age_weeks': [gestational_age]
                })

                category, confidence = predict_birth_weight(model, scaler, label_encoder, input_df, column_order)

                if category is not None:
                    risk_class = "low-risk" if category == "Normal" else "medium-risk" if category == "Low" else "high-risk"

                    st.markdown(f"""
                    <div class="prediction-box {risk_class}">
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
                else:
                    st.error("Prediction failed due to input mismatch or model error.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# === Analytics Tab ===
with tabs[1]:
    st.subheader("Data Insight Dashboard")
    try:
        df = pd.read_csv("analytics_sample.csv")
        st.markdown("### Birth Weight Distribution")
        fig = px.histogram(df, x="birth_weight_category", color="birth_weight_category", title="Weight Categories")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature Correlation")
        corr = df.drop(columns=['birth_weight_category']).corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    except FileNotFoundError:
        st.info("Analytics data not found. Upload `analytics_sample.csv` to see insights.")
    except Exception as e:
        st.error(f"Analytics error: {str(e)}")

# === Footer ===
st.markdown(f"""
---
<div style="text-align: center;">
    <p>© 2025 NexusFlow · Built for SDG 3.1 · <a href="mailto:info@nexusflow.ai">Contact Us</a></p>
    <p><small>Version {APP_VERSION} · Last updated: {pd.Timestamp.now().strftime("%Y-%m-%d")}</small></p>
</div>
""", unsafe_allow_html=True)
