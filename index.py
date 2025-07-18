import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.exceptions import NotFittedError
from streamlit_extras.metric_cards import style_metric_cards

# App config with custom theme and better layout
st.set_page_config(
    page_title="MaternAI: Neonatal Risk Predictor",
    layout="centered",
    page_icon="👶",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main styling */
    .stApp {
        background-color: #fafafa;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 1rem;
    }
    /* Form styling */
    .st-emotion-cache-7ym5gk {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 2rem;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    /* Button styling */
    .st-emotion-cache-7ym5gk .stButton button {
        background-color: #ff7eb9;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    .st-emotion-cache-7ym5gk .stButton button:hover {
        background-color: #ff5d9e;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 94, 158, 0.2);
    }
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        color: #ff7eb9;
    }
    /* Tooltip styling */
    .stTooltip {
        background-color: #333 !important;
        color: white !important;
    }
    /* Metric cards */
    .metric-card {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .st-emotion-cache-7ym5gk {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

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

# Risk interpretation helper
def get_risk_interpretation(category, confidence):
    risk_levels = {
        'Very Low Birth Weight': ('High', '#ff4d4d', 'This indicates significant risk requiring medical attention.'),
        'Low Birth Weight': ('Moderate', '#ffa64d', 'Monitor closely and consider nutritional interventions.'),
        'Normal Birth Weight': ('Low', '#4dff4d', 'Within healthy range, maintain current care.'),
        'High Birth Weight': ('Moderate', '#ffa64d', 'Monitor for potential delivery complications.')
    }
    
    level, color, advice = risk_levels.get(category, ('Unknown', '#cccccc', ''))
    return level, color, advice

# === App Header ===
st.image("https://via.placeholder.com/800x200?text=MaternAI", use_column_width=True)
st.title("👶 MaternAI: Neonatal Risk Predictor")
st.markdown("""
    <div style='background-color: #fff5f7; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;'>
    <p style='color: #333;'>This tool helps predict newborn weight outcomes based on maternal health factors. 
    Complete the form below to assess potential risks and receive personalized insights.</p>
    </div>
""", unsafe_allow_html=True)

# === Progress Bar ===
progress_bar = st.progress(0, text="Form Completion: 0%")

# === User Input Form ===
with st.expander("📋 Maternal Health Information", expanded=True):
    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider(
                "Mother's Age (years)", 
                15, 45, 25,
                help="Maternal age can affect pregnancy outcomes. Teen pregnancies and advanced maternal age (35+) may have higher risks."
            )
            pre_pregnancy_bmi = st.number_input(
                "Pre-pregnancy BMI", 
                10.0, 50.0, 22.0, 0.1,
                help="Body Mass Index before pregnancy. Both underweight (BMI <18.5) and overweight (BMI >25) can impact birth weight."
            )
            gestational_age = st.slider(
                "Gestational Age (weeks)", 
                20, 42, 38,
                help="Duration of pregnancy in weeks. Premature births (<37 weeks) often result in lower birth weights."
            )
            systolic = st.number_input(
                "Systolic BP (mmHg)", 
                80, 200, 110,
                help="Upper number in blood pressure reading. High values may indicate hypertension."
            )
            diastolic = st.number_input(
                "Diastolic BP (mmHg)", 
                50, 130, 70,
                help="Lower number in blood pressure reading. Combined with systolic, indicates cardiovascular health."
            )
            hemoglobin = st.number_input(
                "Hemoglobin (g/dl)", 
                5.0, 18.0, 11.0, 0.1,
                help="Protein in red blood cells that carries oxygen. Low levels may indicate anemia."
            )
            prenatal_visits = st.slider(
                "Prenatal Visits", 
                0, 20, 5,
                help="Number of medical checkups during pregnancy. More visits often correlate with better outcomes."
            )

        with col2:
            diabetes = st.radio(
                "Diabetes", 
                ["Yes", "No"], 
                index=1,
                horizontal=True,
                help="Gestational or pre-existing diabetes can lead to higher birth weights."
            )
            hypertension = st.radio(
                "Hypertension", 
                ["Yes", "No"], 
                index=1,
                horizontal=True,
                help="High blood pressure disorders can affect fetal growth."
            )
            smoking = st.radio(
                "Smoking", 
                ["Yes", "No"], 
                index=1,
                horizontal=True,
                help="Tobacco use is associated with lower birth weights and other complications."
            )
            alcohol = st.radio(
                "Alcohol", 
                ["Yes", "No"], 
                index=1,
                horizontal=True,
                help="Alcohol consumption during pregnancy can impair fetal development."
            )
            education = st.selectbox(
                "Education Level", 
                ["None", "Primary", "Secondary", "Tertiary"],
                help="Higher education levels often correlate with better pregnancy outcomes."
            )
            income = st.selectbox(
                "Household Income", 
                ["Low", "Medium", "High"],
                help="Socioeconomic factors can influence access to prenatal care and nutrition."
            )
            iron = st.radio(
                "Iron Supplements", 
                ["Yes", "No"], 
                index=1,
                horizontal=True,
                help="Iron supplementation helps prevent anemia and supports fetal development."
            )

        # Update progress bar as user interacts with form
        form_elements = [age, pre_pregnancy_bmi, gestational_age, systolic, diastolic, 
                        hemoglobin, prenatal_visits, diabetes, hypertension, smoking, 
                        alcohol, education, income, iron]
        completion = min(100, int(sum(1 for elem in form_elements if (isinstance(elem, (int, float)) and elem != 0) or 
                                  (isinstance(elem, str) and elem in ["Yes", "No", "Primary", "Medium"])) * 7)
        progress_bar.progress(completion, text=f"Form Completion: {completion}%")

        submitted = st.form_submit_button("🔍 Predict Birth Outcome", use_container_width=True)

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

    with st.spinner("🔬 Analyzing maternal health factors..."):
        category, confidence = predict(model, scaler, label_encoder, input_data, feature_order)

        if category:
            st.balloons()
            risk_level, color, advice = get_risk_interpretation(category, confidence)
            
            # Results display
            st.subheader("📊 Prediction Results")
            
            # Metrics cards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {color};">
                    <h3 style="margin: 0; color: #333;">Predicted Category</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0; color: {color};">{category}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {color};">
                    <h3 style="margin: 0; color: #333;">Confidence Level</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0; color: {color};">{confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk visualization
            st.markdown("### 📈 Risk Assessment")
            risk_df = pd.DataFrame({
                'Risk Level': [risk_level],
                'Value': [1]
            })
            
            fig = px.bar(risk_df, x='Value', y='Risk Level', orientation='h', 
                         color_discrete_sequence=[color], height=200)
            fig.update_layout(showlegend=False, xaxis_visible=False, 
                              margin=dict(l=0, r=0, t=0, b=0),
                              plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid {color};'>
                <p style='margin: 0;'>{advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key factors visualization
            st.markdown("### 🔍 Key Influencing Factors")
            feature_importance = pd.DataFrame({
                'Feature': feature_order,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='pinkyl')
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Action buttons
            st.markdown("### 📤 Next Steps")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📥 Save Results", data=input_data.to_csv(), file_name="maternai_prediction.csv")
            with col2:
                if st.button("🖨️ Print Summary"):
                    st.toast("Print function would be implemented in production")
            with col3:
                st.link_button("🩺 Consult Specialist", "https://example.com")
            
        else:
            st.error("Prediction failed. Please check your inputs and try again.")

# === Educational Section ===
st.markdown("---")
with st.expander("📚 Understanding Birth Weight Categories"):
    st.markdown("""
    **Birth weight categories and their significance:**
    
    - **Very Low Birth Weight (<1500g):** High risk for complications including respiratory distress, infections, and developmental delays.
    - **Low Birth Weight (1500-2500g):** Increased risk for health problems, may require special care.
    - **Normal Birth Weight (2500-4000g):** Ideal range with lowest risk of complications.
    - **High Birth Weight (>4000g):** Risk of birth injuries, childhood obesity, and maternal delivery complications.
    
    *Note: These categories are general guidelines. Always consult with your healthcare provider.*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>MaternAI is a predictive tool and not a substitute for professional medical advice.</p>
    <p>© 2023 MaternAI | <a href='#'>Privacy Policy</a> | <a href='#'>Terms of Use</a></p>
</div>
""", unsafe_allow_html=True)
