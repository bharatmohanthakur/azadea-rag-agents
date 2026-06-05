
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Set env vars for LiteLLM/Azure keys
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Set these so Cognee config picks them up if it reads os.environ
deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

os.environ["LLM_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["LLM_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = f"azure/{deployment_name}"

os.environ["EMBEDDING_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["EMBEDDING_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["EMBEDDING_PROVIDER"] = "openai"
os.environ["EMBEDDING_MODEL"] = f"azure/{embedding_deployment}"


import cognee
from cognee.infrastructure.llm.config import get_llm_config, LLMConfig
from cognee.api.v1.search import SearchType

# FORCE update the config in memory
llm_config = get_llm_config()
print(f"Original Config - API Key set: {llm_config.llm_api_key is not None}")
print(f"Original Config - Provider: {llm_config.llm_provider}")

# Manual override to ensure it's set
llm_config.llm_api_key = os.getenv("AZURE_OPENAI_API_KEY")
llm_config.llm_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
llm_config.llm_provider = "openai"
llm_config.llm_model = f"azure/{deployment_name}"

print(f"Updated Config - API Key set: {llm_config.llm_api_key is not None}")
print(f"Updated Config - Model: {llm_config.llm_model}")


async def main():
    print("Testing Cognee with manual config override v6...")
    
    try:
        # Pruning might fail if keys were wrong before, but let's try
        print("Pruning old data...")
        try:
             # Need to ensure system pruning uses the updated config too if it re-renders configs
            await cognee.prune.prune_data() 
            await cognee.prune.prune_system(metadata=True)
        except Exception as e:
            print(f"Prune warning: {e}")

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
