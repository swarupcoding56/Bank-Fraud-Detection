# 🏦 AI-Powered Bank Fraud Detection System

A production-ready machine learning web application for detecting fraudulent bank transactions using XGBoost and Streamlit.

## 🚀 Live Demo

🔗 https://bank-fraud-detection-by-swarup.streamlit.app/

---

# 📌 Project Overview

This project detects fraudulent financial transactions using advanced machine learning techniques on highly imbalanced banking transaction data.

The system includes:

* End-to-end ML pipeline
* Feature engineering
* Imbalance-aware fraud modeling
* Hyperparameter tuning
* Cross-validation
* Real-time prediction using Streamlit

---

# 🧠 Machine Learning Workflow

## ✅ Data Preprocessing

* Missing value handling
* Feature engineering
* Robust Scaling
* Label Encoding

## ✅ Feature Engineering

Created important fraud-detection features such as:

* Log transformed transaction amount
* Balance difference features
* High transaction indicator
* Night transaction detection

## ✅ Imbalanced Data Handling

Used:

* XGBoost scale_pos_weight
* Stratified Cross Validation

without relying heavily on oversampling libraries.

## ✅ Model Training

Compared multiple models:

* Logistic Regression
* Random Forest
* Decision Tree
* KNN
* XGBoost
* Stacking Ensemble

Final selected model:
✅ Tuned XGBoost Classifier

---

# 📊 Model Performance

| Metric          | Score |
| --------------- | ----- |
| ROC-AUC         | ~0.99 |
| Fraud Precision | 97%   |
| Fraud Recall    | 72%   |
| Fraud F1 Score  | 84%   |

The model achieved strong fraud detection performance while maintaining very low false positive rates.

---

# 🌐 Streamlit Web Application

The deployed Streamlit application allows users to:

* Enter transaction details
* Detect fraudulent transactions in real time
* View fraud probability instantly

---

# 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

---

# 📂 Project Structure

```bash
Bank-Fraud-Detection/
│
├── app.py
├── requirements.txt
├── fraud_detection_xgboost.pkl
├── robust_scaler.pkl
├── label_encoder.pkl
├── feature_columns.pkl
├── p99_amount.pkl
└── README.md
```

---

# ▶️ Run Locally

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 👨‍💻 Author

Swarup Mitra

---

# ⭐ Future Improvements

* Real-time transaction streaming
* SHAP explainability integration
* Docker deployment
* API integration using FastAPI
* Cloud deployment pipeline
