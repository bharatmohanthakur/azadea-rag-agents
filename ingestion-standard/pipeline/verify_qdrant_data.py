
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

# Connect to Qdrant
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=api_key)
collection_name = "docs_hybrid_azure_azadea_multimodal"

print(f"Searching collection '{collection_name}' for corrected header...")

# We want to find a chunk containing the merged header
# "OYSHO Pull & Bear" appearing inside a table row/header
target_string = "| OYSHO Pull & Bear |"

# Scroll through points (retrieving text field)
# This is efficient enough for a verification script on a small-ish collection
points, _ = client.scroll(
    collection_name=collection_name,
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="text",
                match=models.MatchText(text="OYSHO Pull & Bear")
            )
        ]
    ),
    limit=10,
    with_payload=True
)

found = False
for point in points:
    text = point.payload.get("text", "")
    if target_string in text:
        print("\n" + "="*50)
        print("✅ SUCCESS: Found corrected data in Qdrant!")
        print("="*50)
        print(f"Point ID: {point.id}")
        print(f"Source: {point.payload.get('source', 'Unknown')}")
        print("-" * 20)
        print("Snippet containing target:")
        
        # Print context around the match
        start_idx = text.find(target_string)
        snippet = text[max(0, start_idx - 100) : min(len(text), start_idx + 200)]
        print(f"...\n{snippet}\n...")
        print("="*50)
        found = True
        break

if not found:
    print("\n❌ FAILURE: Could not find exact string '| OYSHO Pull & Bear |' in returned candidates.")
    print("Checking if old broken format exists...")
    
    # Check for old format
    points_old, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="text",
                    match=models.MatchText(text="OYSHO Pull")
                )
            ]
        ),
        limit=10,
        with_payload=True
    )
    for point in points_old:
        text = point.payload.get("text", "")
        if "| OYSHO Pull |" in text:
             print(f"⚠️ FOUND OLD FORMAT: 'OYSHO Pull |' in {point.payload.get('source')}")

