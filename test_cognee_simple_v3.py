
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Configure environment for LiteLLM (used by Cognee) to use Azure OpenAI
# LiteLLM looks for these specific environment variables for Azure
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Configure Cognee to use the 'openai' provider but point to Azure model
# The model name must start with 'azure/' for LiteLLM to recognize it as Azure OpenAI
deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
os.environ["LLM_PROVIDER"] = "openai" 
os.environ["LLM_MODEL"] = f"azure/{deployment_name}"

# These might be used by Cognee directly or just for explicit fallback
os.environ["LLM_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["LLM_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

import cognee
from cognee.api.v1.search import SearchType

async def main():
    print("Testing Cognee with fixed Azure config...")
    
    # Clean slate
    try:
        print("Pruning old data...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception as e:
        print(f"Prune info: {e}")

    try:
        print("Adding text...")
        await cognee.add("The capital of France is Paris.")
        print("Cognifying...")
        await cognee.cognify()
        print("Searching...")
        results = await cognee.search(SearchType.SIMILARITY, query="capital of France")
        print("Results:")
        for res in results:
            print(res)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
