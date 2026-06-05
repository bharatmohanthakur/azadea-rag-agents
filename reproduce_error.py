
import os
import logging
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reproduction")

# Mock GeneralQueryHandler structure
class GeneralQueryHandler:
    def __init__(self, llm_client, deployment_name):
        self.llm_client = llm_client
        self.deployment_name = deployment_name

    def generate_conversational_response(self, query):
        try:
            messages = [
                {
                    "role": "system", 
                    "content": "You are a friendly HR assistant."
                },
                {
                    "role": "user", 
                    "content": f"User message: {query}"
                }
            ]
            
            print(f"Deployment name: {self.deployment_name}")
            print(f"Client type: {type(self.llm_client)}")
            
            response = self.llm_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    load_dotenv()
    
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    
    print(f"API Key present: {bool(OPENROUTER_API_KEY)}")
    print(f"Model: {OPENROUTER_MODEL}")

    if not OPENROUTER_API_KEY:
        print("Set OPENROUTER_API_KEY env var")
        return

    # Initialize client exactly as in rag_server_gemini.py
    client = wrap_openai(OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    ))

    handler = GeneralQueryHandler(
        llm_client=client,
        deployment_name=OPENROUTER_MODEL
    )

    print("Testing generation...")
    handler.generate_conversational_response("Hello")

if __name__ == "__main__":
    main()
