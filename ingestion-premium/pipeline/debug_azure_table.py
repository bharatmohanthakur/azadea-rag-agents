
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

with open("./data/data/Human Resources/Total Rewards/HRD - TRD - 002 - Uniform Allowance Limits - A - 82.pdf", "rb") as f:
    poller = client.begin_analyze_document(
        "prebuilt-layout",
        body=f,
        features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION]
    )

result = poller.result()

print(f"Detected {len(result.tables)} tables.")

for t_idx, table in enumerate(result.tables):
    print(f"\n--- Table {t_idx} ---")
    for cell in table.cells:
        content = cell.content.replace('\n', ' ')
        if "Pull" in content or "Bear" in content:
            print(f"Row {cell.row_index}, Col {cell.column_index} (span {cell.column_span}): '{content}'")
