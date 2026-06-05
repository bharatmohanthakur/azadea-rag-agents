
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Set standard OpenAI/Azure env vars for LiteLLM
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Configure LLM for Cognee
deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
os.environ["LLM_PROVIDER"] = "openai" 
os.environ["LLM_MODEL"] = f"azure/{deployment_name}"
os.environ["LLM_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY") # Redundant but safe
os.environ["LLM_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")


# Configure Embeddings for Cognee
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
os.environ["EMBEDDING_PROVIDER"] = "openai" # LiteLLM provider
os.environ["EMBEDDING_MODEL"] = f"azure/{embedding_deployment}"
os.environ["EMBEDDING_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["EMBEDDING_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["EMBEDDING_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Manually configure Cognee config
import cognee
from cognee.infrastructure.llm.config import LLMConfig
# Try to force set the config via internal method if available or rely on env vars which seem to be read by get_llm_client

from cognee.api.v1.search import SearchType

async def main():
    print("Testing Cognee with fixed Azure config v5...")
    
    # Clean slate
    try:
        print("Pruning old data...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception as e:
        print(f"Prune info: {e}")

    try:
        print(f"LLM Model: {os.environ.get('LLM_MODEL')}")
        
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
