
import asyncio
import aiohttp
import json

API_URL = "http://localhost:8060/query"
API_RESET_URL = "http://localhost:8060/reset"
USER_ID = "context_loop_test_user"

async def test_loop():
    async with aiohttp.ClientSession() as session:
        # 1. Reset
        print(f"Resetting user {USER_ID}...")
        await session.post(API_RESET_URL, json={"user_id": USER_ID})
        
        # 2. Turn 1: Generic Query
        q1 = "What are the rules for travel expenses?"
        print(f"\nU: {q1}")
        async with session.post(API_URL, json={"query": q1, "user_id": USER_ID}) as resp:
            data = await resp.json()
            print(f"A: {data.get('response')[:200]}...")
            
        # 3. Turn 2: Specific Clarification
        q2 = "I am asking about per diem for international travel."
        print(f"\nU: {q2}")
        async with session.post(API_URL, json={"query": q2, "user_id": USER_ID}) as resp:
            data = await resp.json()
            print(f"A: {data.get('response')[:200]}...")

        # Filler turns to push context
        print("\n--- Inserting filler turns to test history window ---")
        async with session.post(API_URL, json={"query": "Does this apply to managers?", "user_id": USER_ID}) as r: pass
        async with session.post(API_URL, json={"query": "Yes", "user_id": USER_ID}) as r: pass

        # 4. Turn 4: Follow-up (Should maintain context of 'International Per Diem' despite fillers)
        q3 = "What about for local travel?"
        print(f"\nU: {q3}")
        async with session.post(API_URL, json={"query": q3, "user_id": USER_ID}) as resp:
            data = await resp.json()
            r3 = data.get('response')
            print(f"A: {r3[:200]}...")
            
            # Check for loop/loss
            if "I need more information" in r3 or "generic" in r3.lower():
                 print("\n[FAIL] System lost context and went back to generic clarification.")
            else:
                 print("\n[SUCCESS] System seems to have maintained context.")

if __name__ == "__main__":
    asyncio.run(test_loop())
