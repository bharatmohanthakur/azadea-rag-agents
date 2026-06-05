
import os
import glob
import json
import asyncio
from typing import List, Dict, DefaultDict
from collections import defaultdict
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
MD_DIR = "/home/admincsp/multimodal-rag/azadea/md_out_data"
OUTPUT_FILE = "clustered_guidelines.json"

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

def get_cluster_prefix(filename: str) -> str:
    # Example: "HRD - TDD - 010..." -> "HRD - TDD"
    parts = filename.split(" - ")
    if len(parts) >= 2:
        return f"{parts[0]} - {parts[1]}"
    return "MISC" # Default

async def analyze_cluster(cluster_name: str, docs: List[str]) -> Dict:
    """Analyze a cluster of documents to find common context variables."""
    # Create a summary of the cluster contents
    snippets = []
    for doc in docs[:5]: # Take top 5 examples
        with open(doc, "r") as f:
             content = f.read(1000) # First 1000 chars
             snippets.append(f"--- Doc: {os.path.basename(doc)} ---\n{content}\n")
    
    combined_content = "\n".join(snippets)
    
    prompt = f"""You are an expert Policy Analyst.
You are analyzing a CLUSTER of related documents from the category: "{cluster_name}".

Documents Sample:
{combined_content}

Task:
1. Identify the COMMON context variables required to apply policies in this category (e.g. Country, Job Grade, Brand).
2. Formulate a SINGLE, GENERAL guideline for this entire category.

Format:
{{
  "category_description": "Brief description of what this category covers (e.g. HR Training Policies)",
  "condition": "User asks about [Category Description] or related topics",
  "action": "Ask clarifying questions to determine [Common Variables]."
}}

If no variables are seemingly needed, return null.
Response must be valid JSON.
"""
    try:
        response = await client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error processing cluster {cluster_name}: {e}")
        return None

async def main():
    files = list(glob.glob(os.path.join(MD_DIR, "*.md")))
    print(f"Found {len(files)} total documents.")
    
    # 1. Cluster Files
    clusters: DefaultDict[str, List[str]] = defaultdict(list)
    for f in files:
        prefix = get_cluster_prefix(os.path.basename(f))
        clusters[prefix].append(f)
        
    print(f"Identified {len(clusters)} clusters: {list(clusters.keys())[:10]}...")
    
    guidelines = []
    
    # 2. Analyze Each Cluster
    for cluster_name, cluster_docs in clusters.items():
        if len(cluster_docs) < 2: # Skip tiny clusters for now if desired
            continue
            
        print(f"Analyzing Cluster: {cluster_name} ({len(cluster_docs)} docs)...")
        result = await analyze_cluster(cluster_name, cluster_docs)
        
        if result and result.get("category_description"):
             guidelines.append({
                 "cluster": cluster_name,
                 "condition": result["condition"],
                 "action": result["action"]
             })
             print(f"  -> Rule: {result['action']}")

    print(f"\nExtracted {len(guidelines)} clustered guidelines.")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(guidelines, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
