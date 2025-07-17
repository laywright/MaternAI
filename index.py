# app.py - Corrected MaternAI Neonatal Risk Predictor

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
from datetime import datetime
import os

# === 🛠️ CONFIGURATION ===
MODEL_PATH = "model_rf.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
APP_VERSION = "2.3"

# === 🎨 UI CONFIG ===
def setup_ui():
    """Configure UI settings and styles"""
    st.set_page_config(
        page_title="MaternAI | Neonatal Risk Predictor",
        layout="wide",
        page_icon="👶"
    )
    
    st.markdown(f"""
    <style>
        .main {{background-color: #f7f9fc;}}
        h1 {{color: #0072e5;}}
        .stButton>button {{background-color: #0072e5; color: white; border-radius: 0.5em;}}
        .prediction-box {{padding: 1.5em; border-radius: 0.5em; margin: 1em 0;}}
        .low-risk {{background-color: #d4edda; color: #155724;}}
        .medium-risk {{background-color: #fff3cd; color: #856404;}}
        .high-risk {{background-color: #f8d7da; color: #721c24;}}
        .feature-importance {{padding: 1em; background-color: #e9ecef; border-radius: 0.5em;}}
        .info-box {{padding: 1em; background-color: #e7f5ff; border-radius: 0.5em;}}
    </style>
    """, unsafe_allow_html=True)

# === 🧠 MODEL HANDLING ===
def load_model_assets():
    """Load ML model and preprocessing assets with comprehensive error handling"""
    try:
        # Verify files exist and are not empty
        for file_path in [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found: {file_path}")
            if os.path.getsize(file_path) == 0:
                raise ValueError(f"Empty model file: {file_path}")
        
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        
        # Verify model has required attributes
        if not hasattr(model, 'predict_proba'):
            raise AttributeError("Model missing required predict_proba method")
            
        return model, scaler, le
        
    except Exception as e:
        st.error(f"❌ Critical error loading model assets: {str(e)}")
        st.error("Please ensure all model files are properly generated and in the correct location.")
        st.stop()

# === 📊 DATA PROCESSING ===
def prepare_input_data(gestational_age, systolic_bp, diastolic_bp, hemoglobin, parity, 
                      anemia, preeclampsia, alcohol, education):
    """Prepare and validate input data for prediction"""
    try:
        # Process categorical inputs
        parity_val = 5 if parity == "5+" else int(parity)
        anemia_val = 1 if anemia == "Yes" else 0
        preeclampsia_val = 1 if preeclampsia == "Yes" else 0
        alcohol_val = 1 if alcohol == "Yes" else 0
        education_val = ["Primary", "High School", "College", "Postgraduate"].index(education) + 1
        
        # Create input DataFrame with model-expected feature names
        input_data = pd.DataFrame({
            'alcohol_consumption': [alcohol_val],
            'blood_pressure_diastolic': [diastolic_bp],
            'blood_pressure_systolic': [systolic_bp],
            'education_level': [education_val],
            'gestational_age_weeks': [gestational_age],
            'hemoglobin': [hemoglobin],
            'parity': [parity_val],
            'anemia': [anemia_val],
            'preeclampsia': [preeclampsia_val]
        })
        
        # Validate input ranges
        validate_input_ranges(input_data)
        
        return input_data
        
    except ValueError as e:
        st.error(f"Invalid input data: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error preparing data: {str(e)}")
        st.stop()

def validate_input_ranges(input_data):
    """Validate that input values are within expected ranges"""
    validation_rules = {
        'gestational_age_weeks': (15, 45),
        'blood_pressure_systolic': (80, 200),
        'blood_pressure_diastolic': (40, 120),
        'hemoglobin': (5.0, 18.0),
        'parity': (0, 5),
        'anemia': (0, 1),
        'preeclampsia': (0, 1),
        'alcohol_consumption': (0, 1),
        'education_level': (1, 4)
    }
    
    for col, (min_val, max_val) in validation_rules.items():
        if not (min_val <= input_data[col].iloc[0] <= max_val):
            raise ValueError(f"Value for {col} is outside valid range ({min_val}-{max_val})")

# === 📈 VISUALIZATION ===
def show_prediction_result(prediction, confidence, feature_importance=None):
    """Display prediction results with styling"""
    risk_level = "low-risk" if prediction == "Normal" else "medium-risk" if prediction == "Low" else "high-risk"
    
    st.markdown(f"""
    <div class="prediction-box {risk_level}">
        <h3>Prediction Result</h3>
        <p>Predicted Birth Weight Category: <strong>{prediction}</strong></p>
        <p>Confidence: <strong>{confidence:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📊 Detailed Analysis"):
        st.markdown("### Prediction Confidence")
        st.progress(confidence/100)
        
        if feature_importance is not None:
            st.markdown("### Feature Importance")
            fig = px.bar(feature_importance, 
                         x='importance', 
                         y='feature',
                         orientation='h',
                         title="Factors Influencing This Prediction")
            st.plotly_chart(fig, use_container_width=True)
            
        show_prediction_explanation(prediction)

def show_prediction_explanation(prediction):
    """Display explanation of prediction results"""
    st.markdown("### What This Means")
    
    if prediction == "Normal":
        st.info("""
        **Normal birth weight (2500-4000g)**  
        - Typical healthy birth weight range  
        - Lower risk of complications  
        - Standard neonatal care recommended
        """)
    elif prediction == "Low":
        st.warning("""
        **Low birth weight (1500-2500g)**  
        - Increased risk of complications  
        - May require special care after birth  
        - Monitor for feeding difficulties
        """)
    else:
        st.error("""
        **Very low birth weight (<1500g)**  
        - High risk of serious complications  
        - Likely requires NICU care  
        - Increased monitoring needed
        """)

# === 📋 MAIN APP ===
def main():
    setup_ui()
    
    # Title Section
    st.title("👶 MaternAI: Neonatal Risk & Birth Weight Predictor")
    st.markdown(f"<small>Version {APP_VERSION} | Last updated: {datetime.now().strftime('%Y-%m-%d')}</small>", 
               unsafe_allow_html=True)
    
    # Input Section
    st.subheader("📋 Enter Maternal Data")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gestational_age = st.slider("Gestational Age (weeks)", 15, 45, 25, 
                                     help="Maternal age in weeks")
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200, 110,
                                       help="Normal range: 90-120 mmHg")
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 120, 70,
                                        help="Normal range: 60-80 mmHg")
            alcohol = st.radio("Alcohol Consumption", ["No", "Yes"], horizontal=True,
                            help="Any alcohol consumption during pregnancy")
            
        with col2:
            hemoglobin = st.number_input("Hemoglobin Level (g/dl)", 5.0, 18.0, 11.0, step=0.1,
                                      help="Normal range: 12-16 g/dl")
            parity = st.selectbox("Parity (Previous Births)", [0, 1, 2, 3, 4, "5+"],
                                help="Including stillbirths and miscarriages")
            education = st.selectbox("Education Level", 
                                   ["Primary", "High School", "College", "Postgraduate"],
                                   help="Highest education level completed")
            anemia = st.radio("Anemia Present", ["Yes", "No"], horizontal=True)
            preeclampsia = st.radio("History of Preeclampsia", ["Yes", "No"], horizontal=True)
        
        submit = st.form_submit_button("Predict Birth Weight Category")
    
    # Prediction Section
    if submit:
        with st.spinner("🔍 Analyzing data and making prediction..."):
            try:
                # Load model assets
                model, scaler, le = load_model_assets()
                
                # Prepare input data
                input_data = prepare_input_data(
                    gestational_age, systolic_bp, diastolic_bp, hemoglobin,
                    parity, anemia, preeclampsia, alcohol, education
                )
                
                # Scale features
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                decoded = le.inverse_transform([prediction])[0]
                confidence = np.max(model.predict_proba(input_scaled)) * 100
                
                # Get feature importance if available
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': input_data.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                
                # Show results
                show_prediction_result(decoded, confidence, feature_importance)
                
            except Exception as e:
                st.error(f"⚠️ An error occurred during prediction: {str(e)}")
                st.error("Please check your inputs and try again. If the problem persists, contact support.")
    
    # Info Section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        **MaternAI Neonatal Risk Predictor**  
        This tool helps healthcare providers assess the risk of low birth weight  
        based on maternal health indicators.
        
        **How it works:**  
        - Uses machine learning trained on clinical data  
        - Provides risk categorization with confidence scores  
        - Identifies key contributing factors
        
        **Limitations:**  
        - Not a substitute for clinical judgment  
        - Accuracy depends on input data quality  
        - Consult a specialist for high-risk cases
        """)
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center;">
        <p>© 2025 MaternAI · Built for SDG 3.1 · <a href="mailto:info@maternai.org">Contact Us</a></p>
        <p><small>For clinical use only · Not for diagnostic purposes</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
