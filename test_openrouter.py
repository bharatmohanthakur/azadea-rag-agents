
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")

print(f"Testing OpenRouter with model: {model}")
print(f"API Key present: {bool(api_key)}")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

try:
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=10,
        timeout=10.0
    )
    elapsed = time.time() - start
    print(f"Success! Response: {response.choices[0].message.content}")
    print(f"Time taken: {elapsed:.2f}s")
except Exception as e:
    print(f"Error: {e}")
