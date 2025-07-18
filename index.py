import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from sklearn.exceptions import NotFittedError
import plotly.express as px

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="MaternAI - Neonatal Risk Predictor", 
    layout="wide",
    page_icon="🤱",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    :root {
        --primary: #FF7AA2;
        --secondary: #8AB6D6;
        --accent: #FFB8B8;
        --light: #FFF0F5;
        --dark: #4A4A4A;
    }
    
    body {
        background-color: #fafafa;
    }
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .stButton>button {
        border-radius: 12px !important;
        background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        border: none !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(255, 122, 162, 0.2) !important;
    }
    
    .stSelectbox, .stRadio, .stSlider, .stNumberInput {
        background-color: white !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    .card {
        background: white;
        padding: 1.75em;
        margin-bottom: 1.5em;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--secondary), var(--primary)) !important;
    }
    
    .metric-card {
        background: white;
        padding: 1.5em;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .risk-high {
        border-left: 5px solid #FF4D4D;
    }
    
    .risk-medium {
        border-left: 5px solid #FFA64D;
    }
    
    .risk-low {
        border-left: 5px solid #4CAF50;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--dark);
    }
    
    .stMarkdown h1 {
        font-weight: 800;
    }
    
    @media (max-width: 768px) {
        .card {
            padding: 1em;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://images.unsplash.com/photo-1587049352844-4a9a57bced3f", 
             caption="Every mother deserves the best care", 
             use_column_width=True)
with col2:
    st.title("🤱 MaternAI")
    st.markdown("""
    <div style='font-size: 1.2rem; color: var(--dark); margin-bottom: 1.5rem;'>
        Advanced neonatal birth weight prediction for healthier outcomes
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: var(--light); padding: 1rem; border-radius: 12px;'>
        This tool helps predict newborn weight outcomes based on maternal health factors. 
        Complete the form below to assess potential risks and receive personalized insights.
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: var(--primary);'>📋 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 1.5rem;'>
        <h4 style='color: var(--dark); margin-top: 0;'>How to use:</h4>
        <ol style='padding-left: 1.2rem;'>
            <li style='margin-bottom: 0.5rem;'>📝 Fill maternal info</li>
            <li style='margin-bottom: 0.5rem;'>🔎 Click 'Predict'</li>
            <li style='margin-bottom: 0.5rem;'>📊 View results</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
        <h4 style='color: var(--dark); margin-top: 0;'>Resources:</h4>
        <p><a href='https://www.who.int/health-topics/maternal-health' target='_blank'>🌐 WHO Maternal Health</a></p>
        <p><a href='#'>📚 Pregnancy Guidelines</a></p>
        <p><a href='#'>🩺 Find a Specialist</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='margin-top: 2rem; text-align: center; color: var(--dark); font-size: 0.9rem;'>
        Made with ❤️ for mothers worldwide
    </div>
    """, unsafe_allow_html=True)

# --- Load Model Assets ---
@st.cache_resource
def load_assets():
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_order = [
        'age', 'pre_pregnancy_bmi', 'gestational_age_weeks',
        'blood_pressure_systolic', 'blood_pressure_diastolic',
        'hemoglobin_level', 'number_of_prenatal_visits',
        'has_diabetes', 'has_hypertension',
        'smoking_status', 'alcohol_consumption',
        'education_level', 'household_income',
        'iron_supplementation'
    ]
    return model, scaler, label_encoder, feature_order

# --- Predict Function ---
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

# --- Risk Interpretation ---
def get_risk_details(category):
    risk_data = {
        'Very Low Birth Weight': {
            'level': 'High Risk',
            'color': '#FF4D4D',
            'icon': '⚠️',
            'recommendations': [
                "Consult with a maternal-fetal medicine specialist",
                "Consider more frequent prenatal monitoring",
                "Nutritional counseling recommended"
            ]
        },
        'Low Birth Weight': {
            'level': 'Moderate Risk',
            'color': '#FFA64D',
            'icon': '🔍',
            'recommendations': [
                "Schedule additional growth ultrasounds",
                "Monitor nutrition and weight gain",
                "Consider prenatal vitamin adjustments"
            ]
        },
        'Normal Birth Weight': {
            'level': 'Low Risk',
            'color': '#4CAF50',
            'icon': '✅',
            'recommendations': [
                "Continue standard prenatal care",
                "Maintain balanced nutrition",
                "Regular exercise as approved by provider"
            ]
        },
        'High Birth Weight': {
            'level': 'Moderate Risk',
            'color': '#FFA64D',
            'icon': '📈',
            'recommendations': [
                "Monitor for gestational diabetes",
                "Discuss delivery planning with provider",
                "Consider dietary adjustments"
            ]
        }
    }
    return risk_data.get(category, {
        'level': 'Unknown',
        'color': '#CCCCCC',
        'icon': '❓',
        'recommendations': []
    })

# --- Input Form ---
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='color: var(--primary); margin-bottom: 1.5rem;'>
        📋 Maternal Health Information
    </h2>
    """, unsafe_allow_html=True)
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Demographic Information")
            age = st.slider("Mother's Age (years)", 15, 45, 25,
                           help="Maternal age impacts pregnancy outcomes. Teen and advanced maternal age (35+) may have higher risks.")
            pre_pregnancy_bmi = st.number_input("Pre-pregnancy BMI", 10.0, 50.0, 22.0, 0.1,
                                              help="Body Mass Index before pregnancy. Underweight (BMI <18.5) and overweight (BMI >25) can impact birth weight.")
            gestational_age = st.slider("Gestational Age (weeks)", 20, 42, 38,
                                      help="Duration of pregnancy in weeks. Premature births (<37 weeks) often result in lower birth weights.")
            
            st.markdown("#### Health Metrics")
            systolic = st.number_input("Systolic BP (mmHg)", 80, 200, 110,
                                     help="Upper number in blood pressure reading. High values may indicate hypertension.")
            diastolic = st.number_input("Diastolic BP (mmHg)", 50, 130, 70,
                                      help="Lower number in blood pressure reading. Combined with systolic, indicates cardiovascular health.")
            hemoglobin = st.number_input("Hemoglobin Level (g/dl)", 5.0, 18.0, 11.0, 0.1,
                                      help="Protein in red blood cells that carries oxygen. Low levels may indicate anemia.")
            prenatal_visits = st.slider("Number of Prenatal Visits", 0, 20, 5,
                                      help="Recommended at least 8 visits during pregnancy for optimal care.")

        with col2:
            st.markdown("#### Health History")
            diabetes = st.radio("Diabetes History", ["Yes", "No"], horizontal=True,
                              help="Gestational or pre-existing diabetes can lead to higher birth weights.")
            hypertension = st.radio("Hypertension History", ["Yes", "No"], horizontal=True,
                                  help="High blood pressure disorders can affect fetal growth.")
            
            st.markdown("#### Lifestyle Factors")
            smoking = st.radio("Smoking Status", ["Yes", "No"], horizontal=True,
                             help="Tobacco use is associated with lower birth weights and other complications.")
            alcohol = st.radio("Alcohol Consumption", ["Yes", "No"], horizontal=True,
                             help="Alcohol consumption during pregnancy can impair fetal development.")
            
            st.markdown("#### Socioeconomic Factors")
            education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"],
                                   help="Higher education levels often correlate with better pregnancy outcomes.")
            income = st.selectbox("Household Income", ["Low", "Medium", "High"],
                                help="Socioeconomic factors can influence access to prenatal care and nutrition.")
            
            st.markdown("#### Supplementation")
            iron = st.radio("Iron Supplementation", ["Yes", "No"], horizontal=True,
                          help="Iron supplementation helps prevent anemia and supports fetal development.")

        submitted = st.form_submit_button("🚀 Predict Birth Weight", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- On Form Submit ---
if submitted:
    with st.spinner("🔍 Analyzing maternal health data..."):
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
        risk_details = get_risk_details(category)
        
        # Show celebration for good outcomes
        if category == "Normal Birth Weight":
            st.balloons()
        
        # Results Section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"""
        <h2 style='color: var(--primary); margin-bottom: 1rem;'>
            📊 Prediction Results
        </h2>
        """, unsafe_allow_html=True)
        
        # Metrics Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid {risk_details['color']};'>
                <h3 style='margin-top: 0;'>Predicted Category</h3>
                <p style='font-size: 1.8rem; font-weight: 700; margin-bottom: 0; color: {risk_details['color']};'>
                    {risk_details['icon']} {category}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid {risk_details['color']};'>
                <h3 style='margin-top: 0;'>Risk Level</h3>
                <p style='font-size: 1.8rem; font-weight: 700; margin-bottom: 0; color: {risk_details['color']};'>
                    {risk_details['level']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid {risk_details['color']};'>
                <h3 style='margin-top: 0;'>Confidence</h3>
                <p style='font-size: 1.8rem; font-weight: 700; margin-bottom: 0; color: {risk_details['color']};'>
                    {confidence:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence Visualization
        st.markdown("""
        <div style='margin: 1.5rem 0;'>
            <h4 style='margin-bottom: 0.5rem;'>Prediction Confidence</h4>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(confidence))
        
        # Risk Visualization
        st.markdown("""
        <div style='margin: 2rem 0;'>
            <h4 style='margin-bottom: 1rem;'>Risk Assessment</h4>
        </div>
        """, unsafe_allow_html=True)
        
        risk_df = pd.DataFrame({
            'Risk Level': [risk_details['level']],
            'Value': [1],
            'Color': [risk_details['color']]
        })
        
        fig = px.bar(risk_df, x='Value', y='Risk Level', orientation='h',
                     color='Color', color_discrete_map={'Color': risk_details['color']},
                     height=200, text=risk_details['level'])
        fig.update_layout(
            showlegend=False,
            xaxis_visible=False,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(visible=False)
        )
        fig.update_traces(textposition='inside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("""
        <div style='margin: 2rem 0;'>
            <h4 style='margin-bottom: 1rem;'>💡 Recommended Actions</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for rec in risk_details['recommendations']:
            st.markdown(f"""
            <div style='background-color: #FFF9F9; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; 
                        border-left: 4px solid {risk_details['color']};'>
                <p style='margin: 0;'>{rec}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Factors
        st.markdown("""
        <div style='margin: 2rem 0;'>
            <h4 style='margin-bottom: 1rem;'>🔍 Most Influential Factors</h4>
        </div>
        """, unsafe_allow_html=True)
        
        feature_importance = pd.DataFrame({
            'Feature': feature_order,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(5)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='pinkyl',
                     labels={'Feature': '', 'Importance': 'Impact'})
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Disclaimer
        st.markdown("""
        <div style='background-color: #FFF5F5; padding: 1rem; border-radius: 8px; margin-top: 2rem;'>
            <p style='margin: 0; font-size: 0.9rem;'>
                <b>Note:</b> This prediction is a screening tool, not a diagnostic. 
                Please consult a qualified healthcare provider for clinical decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Call to Action
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style='color: var(--primary); margin-bottom: 1.5rem;'>
            📌 Next Steps
        </h3>
        """, unsafe_allow_html=True)
        
        colA, colB, colC = st.columns(3)
        with colA:
            st.button("💾 Save Results", use_container_width=True)
        with colB:
            st.button("📤 Share with Doctor", use_container_width=True)
        with colC:
            st.button("📞 Seek Guidance", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("⚠️ Prediction failed. Please review the input data.")

# --- Educational Section ---
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.expander("📚 Understanding Birth Weight Categories", expanded=False):
        st.markdown("""
        <h4 style='color: var(--dark);'>Birth weight categories and their significance:</h4>
        
        <div style='background-color: #FFF5F5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <h5 style='color: #FF4D4D; margin-top: 0;'>Very Low Birth Weight (<1500g)</h5>
            <p>High risk for complications including respiratory distress, infections, and developmental delays. 
            Requires specialized neonatal care.</p>
        </div>
        
        <div style='background-color: #FFF5F5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <h5 style='color: #FFA64D; margin-top: 0;'>Low Birth Weight (1500-2500g)</h5>
            <p>Increased risk for health problems, may require special care. Associated with higher rates 
            of neonatal morbidity.</p>
        </div>
        
        <div style='background-color: #FFF5F5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <h5 style='color: #4CAF50; margin-top: 0;'>Normal Birth Weight (2500-4000g)</h5>
            <p>Ideal range with lowest risk of complications. Associated with best health outcomes.</p>
        </div>
        
        <div style='background-color: #FFF5F5; padding: 1rem; border-radius: 8px;'>
            <h5 style='color: #FFA64D; margin-top: 0;'>High Birth Weight (>4000g)</h5>
            <p>Risk of birth injuries, childhood obesity, and maternal delivery complications. 
            May indicate gestational diabetes.</p>
        </div>
        
        <p style='font-size: 0.9rem; margin-top: 1rem;'>
            <i>Note: These categories are general guidelines. Always consult with your healthcare provider.</i>
        </p>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div style='text-align: center; color: var(--dark); font-size: 0.9rem; margin-top: 3rem; padding: 1.5rem 0; border-top: 1px solid #f0f0f0;'>
    <p>MaternAI is a predictive tool and not a substitute for professional medical advice.</p>
    <p>© 2025 MaternAI | <a href='#'>Privacy Policy</a> | <a href='#'>Terms of Use</a> | <a href='#'>Contact Us</a></p>
</div>
""", unsafe_allow_html=True)
