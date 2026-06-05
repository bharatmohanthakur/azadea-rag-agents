
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import ingest_data_folder_multimodal as ingest
import azure_doc_intelligence_qdrant as qdrant_ingest

load_dotenv()

# Setup paths
md_file_path = Path("md_out_data_multimodal/HRD - TRD - 002 - Uniform Allowance Limits - A - 82.md").resolve()

print(f"Force ingesting: {md_file_path}")

# Setup Qdrant
qdrant_client = QdrantClient(url=qdrant_ingest.QDRANT_URL, api_key=qdrant_ingest.QDRANT_API_KEY)
dim = qdrant_ingest.infer_embedding_dim()
qdrant_ingest.ensure_collection(qdrant_client, ingest.COLLECTION_NAME_MULTIMODAL, dim)

# Prepare args for ingest_single_md: (md_path, client, collection, idx, total)
args = (md_file_path, qdrant_client, ingest.COLLECTION_NAME_MULTIMODAL, 1, 1)

# Run ingestion
count = ingest.ingest_single_md(args)

print(f"Chunks ingested: {count}")
