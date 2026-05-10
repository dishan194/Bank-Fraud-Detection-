import pandas as pd
import numpy as np

OUTPUT_FILE = "fraud_data.csv"
n_samples = int(input("Enter number of transactions to generate: "))

np.random.seed(42)

# Randomly generate all transactions
data = pd.DataFrame({
    "amount": np.random.uniform(10, 5000, n_samples),
    "transaction_type": np.random.choice(["TRANSFER", "PAYMENT", "DEBIT", "CREDIT"], n_samples),
    "sender_id": np.random.randint(1000, 9999, n_samples),
    "receiver_id": np.random.randint(1000, 9999, n_samples),
    "time_hour": np.random.randint(0, 24, n_samples),
    "old_balance": np.random.uniform(0, 10000, n_samples),
    "new_balance": np.random.uniform(0, 10000, n_samples),
})

# By default 10% fraud, but at least 5 guaranteed fraud transactions
fraud_count = max(5, int(0.1 * n_samples))
fraud_indices = np.random.choice(n_samples, fraud_count, replace=False)

data["isFraud"] = 0
data.loc[fraud_indices, "isFraud"] = 1

# Save to CSV
data.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Dataset '{OUTPUT_FILE}' created successfully with {n_samples} records.")
print(f"💰 Fraudulent transactions: {fraud_count}")

print("\nHere are the first 5 rows:\n")
print(data.head())
