
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Set env vars for Cognee
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = f"azure/{os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'gpt-4.1')}"
os.environ["LLM_ENDPOINT"] = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}openai/deployments/{os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'gpt-4.1')}"
os.environ["LLM_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["LLM_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

import cognee
from cognee.api.v1.search import SearchType

async def main():
    print("Testing Cognee...")
    try:
        print("Adding text...")
        await cognee.add("The capital of France is Paris.")
        print("Cognifying...")
        await cognee.cognify()
        print("Searching...")
        print("Searching...")
        results = await cognee.search(SearchType.CHUNKS, query="capital of France")
        print("Results:", results)
        print("Results:", results)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
