
import requests
import json
import time

BASE_URL = "http://localhost:8060"
USER_ID = "context_test_user_v1"

def ask(query, check_key=None):
    print(f"\n👤 User: {query}")
    start = time.time()
    res = requests.post(f"{BASE_URL}/query", json={"query": query, "user_id": USER_ID})
    
    if res.status_code != 200:
        print(f"Error: {res.text}")
        return None
        
    data = res.json()
    metadata = data.get("metadata", {})
    response = data.get("response", "")
    elapsed = time.time() - start
    
    print(f"🤖 Assistant ({elapsed:.2f}s): {response[:200]}...")
    print(f"   [Complexity: {metadata.get('complexity')}]")
    
    if check_key and check_key in response.lower():
        print(f"   ✅ Found keyword: '{check_key}'")
    
    return data

def main():
    # 0. Reset History
    requests.post(f"{BASE_URL}/reset", json={"user_id": USER_ID})
    
    # 1. Turn 1: Specific question about Pull & Bear
    print("--- Turn 1: Specific Context ---")
    ask("what is the uniform allowance for pull&bear")
    
    # 2. Turn 2: Generic question about Insurance (New Topic)
    print("\n--- Turn 2: Generic Topic (Should trigger Clarification) ---")
    ask("what are the benefits of insurance")
    
    # 3. Turn 3: Short follow-up "Health"
    # This SHOULD mean "Benefits of Health Insurance"
    # User Report: It mixes in "Pull & Bear"
    print("\n--- Turn 3: Ambiguous Follow-up 'Health' ---")
    data = ask("health")
    
    # Check if response mentions "Pull & Bear" (BAD) or just "Health Insurance" (GOOD)
    response = data.get("response", "").lower()
    if "pull" in response or "bear" in response:
        print("\n❌ FAIL: Response contains 'Pull & Bear' context contamination!")
    else:
        print("\n✅ PASS: Response seems focused on Health Insurance only.")

if __name__ == "__main__":
    main()
