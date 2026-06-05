
import asyncio
import httpx
import time
import json
import statistics

URL = "http://localhost:7071/query"
CONCURRENT_USERS = 10
QUERY = "What are the maternity leave policies for UAE?"

async def simulate_user(user_id):
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "query": QUERY,
            "user_id": f"stress_test_user_{user_id}"
        }
        start_time = time.time()
        try:
            response = await client.post(URL, json=payload)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                return {"status": "success", "user_id": user_id, "time": elapsed}
            else:
                return {"status": "error", "user_id": user_id, "time": elapsed, "code": response.status_code, "msg": response.text}
        except Exception as e:
            elapsed = time.time() - start_time
            return {"status": "exception", "user_id": user_id, "time": elapsed, "msg": str(e)}

async def main():
    print(f"🚀 Starting stress test on {URL}")
    print(f"👥 Users: {CONCURRENT_USERS}")
    print(f"❓ Query: {QUERY}\n")
    
    start_total = time.time()
    
    tasks = [simulate_user(i) for i in range(CONCURRENT_USERS)]
    results = await asyncio.gather(*tasks)
    
    end_total = time.time()
    total_time = end_total - start_total
    
    # Analysis
    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] != "success"]
    
    times = [r["time"] for r in successes]
    
    print("\n" + "="*40)
    print("📊 RESULTS")
    print("="*40)
    print(f"Total Time:       {total_time:.2f}s")
    print(f"Successful Req:   {len(successes)}/{CONCURRENT_USERS}")
    print(f"Failed Req:       {len(errors)}/{CONCURRENT_USERS}")
    
    if times:
        print(f"Avg Response:     {statistics.mean(times):.2f}s")
        print(f"Min Response:     {min(times):.2f}s")
        print(f"Max Response:     {max(times):.2f}s")
        
    if errors:
        print("\n❌ ERRORS:")
        for e in errors:
            print(f"User {e['user_id']}: {e.get('code', 'N/A')} - {e.get('msg', '')}")

if __name__ == "__main__":
    asyncio.run(main())
