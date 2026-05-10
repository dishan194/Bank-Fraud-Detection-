import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# -------------------------
# CONFIG
# -------------------------
OUTPUT_DIR = "output"
MODEL_FILE = os.path.join(OUTPUT_DIR, "fraud_model.joblib")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# GENERATE SYNTHETIC DATA
# -------------------------
n_samples = 5000
np.random.seed(42)

data = pd.DataFrame({
    "amount": np.random.uniform(10, 5000, n_samples),
    "transaction_type": np.random.choice(["TRANSFER", "PAYMENT", "DEBIT", "CREDIT"], n_samples),
    "sender_id": np.random.randint(1000, 9999, n_samples),
    "receiver_id": np.random.randint(1000, 9999, n_samples),
    "time_hour": np.random.randint(0, 24, n_samples),
    "old_balance": np.random.uniform(0, 10000, n_samples),
    "new_balance": np.random.uniform(0, 10000, n_samples),
    "isFraud": np.random.choice([0,1], n_samples, p=[0.9,0.1])
})

# -------------------------
# ENCODE CATEGORICAL COLUMN
# -------------------------
le = LabelEncoder()
data["transaction_type"] = le.fit_transform(data["transaction_type"])

# -------------------------
# SPLIT FEATURES & TARGET
# -------------------------
X = data.drop("isFraud", axis=1)
y = data["isFraud"]

# -------------------------
# BALANCE DATA
# -------------------------
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print(f"Balanced classes: {np.bincount(y_res)}")

# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# -------------------------
# SCALE FEATURES
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -------------------------
# TRAIN MODEL
# -------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model trained successfully!")

# -------------------------
# SAVE MODEL
# -------------------------
joblib.dump({"model": model, "scaler": scaler, "encoder": le}, MODEL_FILE)
print(f"Model saved at {MODEL_FILE}")
