
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Explicitly set Azure OpenAI config for Cognee
# Cognee uses 'openai' provider but with specific endpoint/model for Azure
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = f"azure/{os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'gpt-4.1')}"
os.environ["LLM_ENDPOINT"] = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}" # Cognee might append openai/deployments... let's check docs or try standard azure endpoint format
os.environ["LLM_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["LLM_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Also set standard OpenAI vars as fallback/if litellm needs them
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")


import cognee
from cognee.api.v1.search import SearchType

async def main():
    print("Testing Cognee with fixed config...")
    
    # Prune to start fresh
    print("Pruning old data...")
    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception as e:
        print(f"Prune warning: {e}")

    try:
        print("Adding text...")
        await cognee.add("The capital of France is Paris.")
        print("Cognifying...")
        await cognee.cognify()
        print("Searching...")
        results = await cognee.search(SearchType.SIMILARITY, query="capital of France")
        print("Results:", results)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
