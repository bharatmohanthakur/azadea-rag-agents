
import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(host=qdrant_host, port=qdrant_port, api_key=api_key)
collection_name = "docs_hybrid_azure_azadea_multimodal"

target_file = "HRD - TRD - 002 - Uniform Allowance Limits - A - 82.md"

print(f"Deleting duplicates for {target_file} from '{collection_name}'...")

# 1. Delete points for this file
client.delete(
    collection_name=collection_name,
    points_selector=models.FilterSelector(
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=target_file)
                )
            ]
        )
    )
)
print("Deletion command sent.")

# 2. Verify deletion
print("Verifying deletion...")
points, _ = client.scroll(
    collection_name=collection_name,
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="source_file",
                match=models.MatchValue(value=target_file)
            )
        ]
    ),
    limit=5
)

if not points:
    print("✅ Verified: All chunks for this file are deleted.")
else:
    print(f"❌ Error: Found {len(points)} points remaining!")
    sys.exit(1)
