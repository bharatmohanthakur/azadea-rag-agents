
import os
import glob
import json
import asyncio
from typing import List, Dict
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
MD_DIR = "/home/admincsp/multimodal-rag/azadea/md_out_data"
OUTPUT_FILE = "inferred_guidelines.json"

# specific client setup from existing code
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o")

client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

async def analyze_doc(filename: str, content: str) -> Dict:
    """Asks LLM to extract guidelines from a single doc."""
    prompt = f"""You are an expert Policy Analyst.
Analyze the following document to identify any CONTEXT VARIABLES required to answer questions about it.

Document: {filename}
Content Snippet:
{content[:8000]}...

Does this policy depend on factors like:
- Country/Location?
- Job Grade/Role?
- Brand/Department?
- Marital Status?

If YES, formulate a clear behavioral guideline for an AI agent.
Format:
{{
  "topic": "Brief topic name (e.g. Maternity Leave)",
  "condition": "User asks about [Topic]",
  "action": "Ask clarifying questions to determine [Variables]."
}}

If the document applies UNIVERSALLY (no variables needed), return null.
Response must be valid JSON.
"""
    try:
        response = await client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

async def main():
    files = list(glob.glob(os.path.join(MD_DIR, "*.md")))
    print(f"Found {len(files)} documents.")
    
    # Process a subset or all? Let's do a sample of 10 diverse files to show the user.
    # In production we'd do all.
    sample_files = files[:10] 
    
    guidelines = []
    
    for filepath in sample_files:
        filename = os.path.basename(filepath)
        print(f"Analyzing {filename}...")
        with open(filepath, "r") as f:
            content = f.read()
            
        if len(content) < 500:
            continue
            
        result = await analyze_doc(filename, content)
        if result and result.get("topic"):
            # Clean up the output to be Parlant-ready
            guideline = {
                "source_doc": filename,
                "condition": f"User asks about {result['topic']}",
                "action": result['action']
            }
            guidelines.append(guideline)
            print(f"  -> Generated: {guideline['action']}")

    print(f"\nExtracted {len(guidelines)} guidelines.")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(guidelines, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
