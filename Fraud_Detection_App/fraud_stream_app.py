import streamlit as st
from kafka import KafkaConsumer
import json
import joblib
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(page_title="Real-Time Fraud Detection System", layout="wide")

st.markdown("""
    <h1 style='text-align:center;color:#00C4CC;'>💳 Real-Time Bank Payment Fraud Detection</h1>
    <h4 style='text-align:center;'>Using Apache Kafka Streaming + ML Model</h4>
""", unsafe_allow_html=True)

# Load your trained model and scaler
model = joblib.load("output/fraud_model.joblib")
scaler = joblib.load("output/scaler.joblib")

# Kafka Consumer Setup
consumer = KafkaConsumer(
    'fraud-transactions',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='fraud-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# ---------------- DATA STORAGE ----------------
transactions = []
fraud_count = 0
safe_count = 0
total_amount = 0.0

# Streamlit layout containers
placeholder = st.empty()
col1, col2, col3 = st.columns(3)

fraud_placeholder = col1.metric("🚨 Fraudulent Transactions", 0)
safe_placeholder = col2.metric("✅ Safe Transactions", 0)
amount_placeholder = col3.metric("💰 Total Amount Processed", "₹0.00")

chart_placeholder = st.empty()

# ---------------- STREAM PROCESSING ----------------
for message in consumer:
    data = message.value

    # Feature extraction
    features = np.array([[data['amount'], data['time_hour'],
                          data['sender_balance'], data['receiver_balance'],
                          1 if data['type'] == "Person-to-Merchant" else 0]])
    
    X_scaled = scaler.transform(features)
    prediction = model.predict(X_scaled)[0]

    result = "FRAUD" if prediction == 1 else "SAFE"
    transactions.append({
        "Sender": data['sender'],
        "Receiver": data['receiver'],
        "Amount": data['amount'],
        "Type": data['type'],
        "Result": result
    })

    # Update counters
    if result == "FRAUD":
        fraud_count += 1
    else:
        safe_count += 1
    total_amount += data['amount']

    # Convert to DataFrame
    df = pd.DataFrame(transactions[-50:])  # show last 50 transactions

    # Display metrics
    fraud_placeholder.metric("🚨 Fraudulent Transactions", fraud_count)
    safe_placeholder.metric("✅ Safe Transactions", safe_count)
    amount_placeholder.metric("💰 Total Amount Processed", f"₹{total_amount:,.2f}")

    # Display latest transactions table
    with placeholder.container():
        st.markdown("### 🧾 Live Transaction Stream")
        st.dataframe(df, use_container_width=True)

    # Update live pie chart
    fig, ax = plt.subplots()
    labels = ['Fraud', 'Safe']
    sizes = [fraud_count, safe_count]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#FF4B4B', '#4BFF85'])
    ax.axis('equal')
    chart_placeholder.pyplot(fig)

    time.sleep(1)
