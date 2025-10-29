**Planning README structure**

The user wants a complete README in markdown, and it needs to include installation commands, requirements (maybe specific versions), and the author's name — Auchitya. I’ll keep it concise but complete. I'll use clear headings, up to 6-8 structured sections, making sure to include any run instructions and features. I think I’ll also include a section on model choice and project structure in a code block. I just need to make sure not to make it too long. Alright, let’s keep it neat and organized.
```markdown
# 💳 Fraud Detection Web App

An end-to-end machine learning project for fraud detection, built by **Auchitya**.  
Includes data preprocessing, feature engineering, model training and comparison, and a Streamlit web app for real-time predictions.

---

## 📦 What’s inside
- **EDA & Feature engineering:** log transforms, time features, distance, risk flags
- **Models compared:** Logistic Regression, Random Forest, XGBoost
- **Chosen model:** Random Forest (best balance of precision and recall)
- **Web app:** Streamlit app that computes engineered features in the background and predicts Fraud / Not Fraud

---

## 📂 Project structure
```
├── app.py                       # Streamlit web app
├── fraud_detection_rf_model.pkl # Saved Random Forest pipeline (joblib)
├── requirements.txt             # Pinned Python dependencies
├── notebook.ipynb               # EDA, feature engineering, model training
└── README.md                    # Project documentation
```

---

## ⚙️ Installation
1. Clone the repo
   ```bash
   git clone https://github.com/yourusername/fraud-detection-app.git
   cd fraud-detection-app
   ```

2. Create and activate a virtual environment (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

Pinned versions used:
```
streamlit==1.39.0
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
xgboost==2.1.1
joblib==1.4.2
matplotlib==3.9.2
seaborn==0.13.2
```

---

## 🚀 Run the app
```bash
streamlit run app.py
```
Open the local URL shown in the terminal (usually http://localhost:8501).

---

## 🧠 How the app works
- User enters transaction details (amount, location, time, category, etc.)
- App computes engineered features (log amounts, weekend flag, distance, risk score)
- The saved **Random Forest pipeline** predicts Fraud / Not Fraud and shows probability
- Transparent view of computed features available via an expander panel

---

## 📊 Model comparison (fraud class)
- **Logistic Regression:** Precision ~0.03, Recall ~0.77, F1 ~0.06 → catches many but too many false alarms
- **Random Forest:** Precision ~0.86, Recall ~0.62, F1 ~0.72 → best practical balance ✅
- **XGBoost:** Precision ~0.34, Recall ~0.89, F1 ~0.49 → high recall, lower precision

Chosen model: **Random Forest** for production due to high precision and solid recall.

---

## 👨‍💻 Author
Project created by **Auchitya** for a professional data science portfolio.  
If you use this project or have feedback, feel free to reach out.
```
