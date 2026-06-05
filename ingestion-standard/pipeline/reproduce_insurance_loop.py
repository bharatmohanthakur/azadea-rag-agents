
import asyncio
import aiohttp
import json
import sys

API_URL = "http://localhost:8060/query"
API_RESET_URL = "http://localhost:8060/reset"
USER_ID = "insurance_loop_tester"

async def test_loop():
    async with aiohttp.ClientSession() as session:
        # Reset
        print(f"--- Resetting User {USER_ID} ---")
        await session.post(API_RESET_URL, json={"user_id": USER_ID})
        
        # Turn 1: Generic
        q1 = "How can I benefit from insurance?"
        print(f"\nU1: {q1}")
        async with session.post(API_URL, json={"query": q1, "user_id": USER_ID}) as resp:
            d1 = await resp.json()
            print(f"A1: {d1.get('response')[:300]}...")
            
        # Turn 2: Answer Clarification (Health)
        q2 = "Health"
        print(f"\nU2: {q2}")
        async with session.post(API_URL, json={"query": q2, "user_id": USER_ID}) as resp:
            d2 = await resp.json()
            print(f"A2: {d2.get('response')[:300]}...")
            
            if "health" in d2.get('response').lower() and "insurance" in d2.get('response').lower():
                 print("[INFO] Turn 2 acknowledged 'Health'.")
            else:
                 print("[WARN] Turn 2 did not seem to acknowledge 'Health'. Potential Loop.")

        # Turn 3: Deepen (e.g. In-patient or Class A) - simulating user stuck
        q3 = "In-patient"
        print(f"\nU3: {q3}")
        async with session.post(API_URL, json={"query": q3, "user_id": USER_ID}) as resp:
            d3 = await resp.json()
            print(f"A3: {d3.get('response', '')[:300]}...")

if __name__ == "__main__":
    asyncio.run(test_loop())
