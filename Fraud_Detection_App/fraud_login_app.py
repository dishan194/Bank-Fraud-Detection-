import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_DIR = "output"
USER_CREDENTIALS = {
    "admin": "1234",
    "user": "pass"
}

st.set_page_config(page_title="Fraud Detection App", layout="centered")

# Load latest saved model
def load_latest_model():
    if not os.path.exists(MODEL_DIR):
        return None, None
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
    if not models:
        return None, None
    latest = sorted(models)[-1]
    return joblib.load(os.path.join(MODEL_DIR, latest)), latest

model_obj, model_name = load_latest_model()
if model_obj is None:
    st.error("⚠️ No trained model found! Please run fraud_detection.py first to train and save a model.")
    st.stop()

model = model_obj["model"]
scaler = model_obj["scaler"]

st.success(f"✅ Loaded model: {model_name}")

# -------------------
# LOGIN PAGE
# -------------------
def login():
    st.title("🔐 Bank Fraud Detection - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.success("✅ Login successful!")
        else:
            st.error("❌ Invalid username or password")

# -------------------
# MAIN APP
# -------------------
def app():
    st.title("💳 Fraud Detection System")
    st.write("Enter payment transaction details:")

    # Real payment fields
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "PAYMENT", "DEBIT", "CREDIT"])
    sender_id = st.number_input("Sender ID", min_value=1000, max_value=9999, value=1234)
    receiver_id = st.number_input("Receiver ID", min_value=1000, max_value=9999, value=5678)
    time_hour = st.slider("Transaction Hour (0–23)", 0, 23, 12)
    old_balance = st.number_input("Old Balance", min_value=0.0, value=2000.0)
    new_balance = st.number_input("New Balance", min_value=0.0, value=1500.0)

    if st.button("Predict Fraud"):
        # Convert transaction type to numeric for model
        transaction_mapping = {"TRANSFER":0, "PAYMENT":1, "DEBIT":2, "CREDIT":3}
        trans_encoded = transaction_mapping[transaction_type]

        features = np.array([[amount, trans_encoded, sender_id, receiver_id,
                              time_hour, old_balance, new_balance]])
        X_scaled = scaler.transform(features)
        proba = model.predict_proba(X_scaled)[:,1][0]
        pred = int(proba >= 0.5)

        if pred == 1:
            st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Fraud Probability: {proba:.2f})")

    # Logout button
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

# -------------------
# APP ENTRY POINT
# -------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    app()
