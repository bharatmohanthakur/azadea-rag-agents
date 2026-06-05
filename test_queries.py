
import requests
import time
import json

queries = [
    "what is the dress code?",
    "how to apply for leave?",
    "tell me about salary structure",
    "who is the CEO?",
    "thanks bye",
    "i am frustrated with the system",
    "medical insurance details",
    "work from home policy",
    "shift timings"
]

url = "http://localhost:8088/query"

for i, q in enumerate(queries):
    print(f"[{i+1}/{len(queries)}] Sending: '{q}'...")
    try:
        response = requests.post(
            url, 
            json={"query": q, "user_id": "stress_test"},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed: {e}")
    time.sleep(2) # Brief pause between queries
