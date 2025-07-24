MaternAI 🧠🤰🏿
An AI-powered maternal health tool that predicts whether a newborn is likely to be underweight or full-weight at birth—designed for expectant mothers, clinicians, and public health institutions in Sub-Saharan Africa.

🌍 Overview
MaternAI leverages machine learning to identify risk factors contributing to low birth weight. The platform supports early intervention and empowers healthcare workers and mothers with timely, data-driven insights.

🔧 Features
🧮 Predicts birth weight category based on demographic & clinical inputs

📊 Easy-to-use Streamlit interface for healthcare providers

📱 Mobile-first design for low-resource settings

🌐 Localization for Kenya (can be adapted to other regions)

🔐 Privacy-aware: no personal data is stored

🚀 How It Works
User inputs maternal and pregnancy-related parameters

Model processes inputs using a trained classifier

Result: Prediction (Likely Underweight / Full Weight) with confidence score

Recommendations displayed based on risk level

🛠️ Tech Stack
Python

Streamlit

Pandas & Scikit-learn

Matplotlib / Plotly (for visualizations)

GitHub Actions (for CI/CD – optional)

🧪 Run Locally
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/MaternAI.git
cd MaternAI
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Launch app:

bash
Copy
Edit
streamlit run app.py
📈 Dataset
MaternAI was trained on anonymized health datasets reflecting key maternal health parameters in Kenya and surrounding regions. All data is preprocessed and embedded within the training pipeline.

Note: For privacy reasons, the raw datasets are not included in this repo. Contact the team for a sample version.

💡 Project Goals
Improve early detection of at-risk pregnancies

Support maternal care planning with digital tools

Reduce neonatal mortality rates through preventative alerts

🤝 Contributors
Muiruri Alex — Machine Learning & Streamlit

[Add Other Contributors Here]

📫 Contact
Want to collaborate or pilot MaternAI at your facility?
Email us at: maternai.health@gmail.com
Or visit: [YourWebsiteIfAny.com]

⚖️ License
This project is licensed under the MIT License — see the LICENSE file for details.
