
import asyncio
import argparse
import json
import httpx
import sys

# Script to verify the /query endpoint streaming behavior
# Usage: python3 verify_real_streaming.py --query "What is the maternity leave policy?"

async def test_streaming(query):
    url = "http://localhost:8043/query/stream"
    payload = {
        "query": query,
        "user_id": "test_user_verify"
    }
    
    print(f"🚀 Sending query to {url}: {query}")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                print(f"❌ Error: Status code {response.status_code}")
                content = await response.read()
                print(content.decode())
                sys.exit(1)
                
            print("✅ Connection established. Receiving stream...")
            
            chunk_count = 0
            token_count = 0
            sources_found = 0
            
            async for line in response.aiter_lines():
                if not line or not line.strip():
                    continue
                    
                if line.startswith("data: "):
                    json_str = line[6:]
                    try:
                        data = json.loads(json_str)
                        msg_type = data.get("type")
                        
                        if msg_type == "token":
                            print(f"T", end="", flush=True) # visual indicator of token
                            token_count += 1
                        elif msg_type == "source_found":
                            print(f"\n[SOURCE: {data.get('source')}]")
                            sources_found += 1
                        elif msg_type == "status":
                            print(f"\n[STATUS: {data.get('message')}]")
                        elif msg_type == "progress":
                            print(f"\n[PROGRESS: {data.get('percentage')}% - {data.get('message')}]")
                        elif msg_type == "done":
                            print(f"\n✅ DONE. Metadata keys: {list(data.get('metadata', {}).keys())}")
                        elif msg_type == "error":
                            print(f"\n❌ ERROR: {data.get('error')}")
                        
                        chunk_count += 1
                        
                    except json.JSONDecodeError:
                        print(f"\n⚠️ Invalid JSON: {line}")
            
            print(f"\n\n📊 Summary:")
            print(f"   - Chunks received: {chunk_count}")
            print(f"   - Tokens streamed: {token_count}")
            print(f"   - Sources found: {sources_found}")
            
            if token_count > 0:
                print("✅ TEST PASSED: Tokens were streamed.")
            else:
                print("❌ TEST FAILED: No tokens received.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What is the split of sick leave days?", help="Query to test")
    args = parser.parse_args()
    
    try:
        asyncio.run(test_streaming(args.query))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"\n❌ Exception: {e}")
