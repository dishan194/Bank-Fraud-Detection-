import streamlit as st
import joblib
import pandas as pd

st.markdown("<h2 class='title-text'>📝 Manual Transaction Entry</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Enter custom transaction data to test the fraud detection model.</p>", unsafe_allow_html=True)

# Load model + scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

transaction_types = ["DEBIT", "CREDIT", "PAYMENT", "TRANSFER", "WITHDRAWAL"]

with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    # Input fields (matching dataset columns)
    amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=1000.0, step=100.0)
    transaction_type = st.selectbox("Transaction Type", transaction_types)
    sender_id = st.number_input("Sender ID", min_value=0, step=1)
    receiver_id = st.number_input("Receiver ID", min_value=0, step=1)
    time_hour = st.slider("Hour of Transaction (0–23)", 0, 23, 12)

    old_balance = st.number_input("Old Balance", min_value=0.0, value=5000.0)
    new_balance = st.number_input("New Balance", min_value=0.0, value=6000.0)

    st.write("---")
    if st.button("Predict Fraud Status", use_container_width=True):

        # Convert category to numeric (label encoding style)
        type_mapping = {
            "DEBIT": 0,
            "CREDIT": 1,
            "PAYMENT": 2,
            "TRANSFER": 3,
            "WITHDRAWAL": 4
        }

        encoded_type = type_mapping[transaction_type]

        # Create dataframe for prediction
        input_df = pd.DataFrame([{
            "amount": amount,
            "transaction_type": encoded_type,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "time_hour": time_hour,
            "old_balance": old_balance,
            "new_balance": new_balance,
            "isFraud": 0   # placeholder; model predicts fraud
        }])

        # The scaler expects ALL numerical columns
        scaled = scaler.transform(input_df)

        prediction = model.predict(scaled)[0]

        if prediction == 1:
            st.error("⚠️ Model Prediction: **FRAUDULENT TRANSACTION**")
        else:
            st.success("✅ Model Prediction: **SAFE TRANSACTION**")

    st.markdown("</div>", unsafe_allow_html=True)
