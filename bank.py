import streamlit as st

import pandas as pd

import numpy as np

import joblib

# ======================================
# LOAD OBJECTS
# ======================================

model = joblib.load(
    "fraud_detection_xgboost.pkl"
)

scaler = joblib.load(
    "robust_scaler.pkl"
)

le = joblib.load(
    "label_encoder.pkl"
)

feature_columns = joblib.load(
    "feature_columns.pkl"
)
p99_amount=joblib.load("p99_amount.pkl"
)
# ======================================
# PREPROCESS FUNCTION
# ======================================

def preprocess_input(data):

    data["log_amount"] = np.log1p(
        data["amount"]
    )

    data["high_transaction"] = (
    data["amount"] > p99_amount
).astype(int)
    data["balanced_org"] = (
        data["oldbalanceOrg"] -
        data["newbalanceOrig"]
    )

    data["balanced_dest"] = (
        data["newbalanceDest"] -
        data["oldbalanceDest"]
    )

    data["type_encoded"] = le.transform(
        data["type"]
    )
   # EXTRA FEATURES
    data["isFlaggedFraud"] = 0

    data["hour"] = data["step"] % 24

    data["night"] = (
    (data["hour"] >= 22) |
    (data["hour"] <= 6)
).astype(int)
    data.drop(
        columns=[
            "type",
            "oldbalanceOrg",
            "newbalanceOrig",
            "newbalanceDest",
            "oldbalanceDest"
        ],
        inplace=True
    )

    data = data[feature_columns]

    scale_columns = [

        "step",

        "log_amount",

        "balanced_org",

        "balanced_dest"
    ]

    data[scale_columns] = scaler.transform(
        data[scale_columns]
    )

    return data

# ======================================
# PREDICTION FUNCTION
# ======================================

def predict_fraud(data):

    processed_data = preprocess_input(
        data.copy()
    )

    prediction = model.predict(
        processed_data
    )[0]

    probability = model.predict_proba(
        processed_data
    )[0][1]

    return prediction, probability

# ======================================
# STREAMLIT UI
# ======================================

st.title("Bank Fraud Detection System")

st.write(
    "Enter Transaction Details"
)

# ======================================
# USER INPUTS
# ======================================

step = st.number_input(
    "Step",
    min_value=1
)

transaction_type = st.selectbox(
    "Transaction Type",

    [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]
)

amount = st.number_input(
    "Amount",
    min_value=0.0
)

oldbalanceOrg = st.number_input(
    "Old Balance Origin",
    min_value=0.0
)

newbalanceOrig = st.number_input(
    "New Balance Origin",
    min_value=0.0
)

oldbalanceDest = st.number_input(
    "Old Balance Destination",
    min_value=0.0
)

newbalanceDest = st.number_input(
    "New Balance Destination",
    min_value=0.0
)

# ======================================
# PREDICT BUTTON
# ======================================

if st.button("Predict Fraud"):

    input_data = pd.DataFrame({

        "step": [step],

        "type": [transaction_type],

        "amount": [amount],

        "oldbalanceOrg": [oldbalanceOrg],

        "newbalanceOrig": [newbalanceOrig],

        "oldbalanceDest": [oldbalanceDest],

        "newbalanceDest": [newbalanceDest]
    })

    prediction, probability = predict_fraud(
        input_data
    )

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("Fraud Transaction Detected")

    else:

        st.success("Normal Transaction")

    st.write(
        f"Fraud Probability : {probability:.4f}"
    )