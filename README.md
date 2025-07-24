# MaternAI 🤰🏿🧠

MaternAI is an AI-powered maternal health prediction tool that forecasts whether a newborn is likely to be born underweight or at full weight based on maternal health and demographic data. It is designed for use by health professionals, clinics, and mothers, with a focus on accessibility in Sub-Saharan Africa.

---

## 🌍 Overview

MaternAI leverages machine learning to analyze key maternal indicators and provide early warning for low birth weight (LBW) outcomes. This allows for proactive care and intervention, especially in underserved communities.

---

## 🔧 Features

- ✅ Predicts birth weight category using clinically relevant inputs  
- 📊 User-friendly Streamlit web interface  
- 📱 Mobile-responsive UI for use in low-resource settings  
- 🔐 Privacy-conscious – no sensitive data is stored or shared  
- 🌐 Customizable for different regional datasets

---

## 🚀 How It Works

1. Users input maternal data (e.g. age, education, weight, prenatal visits)
2. The backend model processes the data using a trained ML classifier
3. The app returns a prediction (Full Weight or Underweight) with a confidence score
4. Actionable insights and recommendations are displayed

---

## 🛠️ Built With

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- Plotly / Matplotlib (for visualization)

---

## 📦 Installation

To run MaternAI locally:

```bash
git clone https://github.com/yourusername/MaternAI.git
cd MaternAI
pip install -r requirements.txt
streamlit run app.py
