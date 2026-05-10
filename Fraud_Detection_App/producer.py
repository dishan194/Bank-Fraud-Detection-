from kafka import KafkaProducer
import json
import time
import random

# Create Kafka producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("🚀 Sending live UPI transactions...")

while True:
    transaction = {
        "transaction_id": random.randint(100000, 999999),
        "sender": f"user{random.randint(1, 20)}@upi",
        "receiver": f"merchant{random.randint(1, 10)}@upi",
        "amount": round(random.uniform(50, 20000), 2),
        "time_hour": random.randint(0, 23),
        "sender_balance": round(random.uniform(1000, 100000), 2),
        "receiver_balance": round(random.uniform(100, 50000), 2),
        "type": random.choice(["Person-to-Person", "Person-to-Merchant"])
    }

    producer.send("transactions", value=transaction)
    print("📤 Sent:", transaction)

    time.sleep(2)   # send every 2 seconds
