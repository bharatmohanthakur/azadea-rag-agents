#!/usr/bin/env python3
"""
MCP server exposing the Azadea OCI knowledge base (live Qdrant collection
`docs_oci_ingested_azadea`) to Claude Code.

It reuses the SAME retrieval the production OCI agent uses
(`_tool_get_document_knowledge`): OCI Cohere v4 dense embedding + BM25 sparse,
RRF hybrid fusion, plus neighbor table_summary expansion. That's why it works
against the existing collection — a generic Qdrant MCP server would embed
queries with a different model and never match these vectors.

Register with:
    claude mcp add azadea-kb -- \
        /home/admincsp/multimodal-rag/azadea/.venv/bin/python3 \
        /home/admincsp/graphiti_fixed_test/ingestion-oci/mcp_azadea_qdrant.py
"""
import os
import sys
import json

# Make the ingestion-oci modules importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from agent_tools import _tool_get_document_knowledge, COLLECTION_NAME

mcp = FastMCP("azadea-kb")


@mcp.tool()
def search_azadea_knowledge(query: str) -> str:
    """Search the Azadea Group policy/procedure knowledge base (HR, Finance, IT,
    Operations, Stock, F&B, etc.). Returns the most relevant document chunks —
    including full table contents (refund periods, discount/insurance matrices,
    salary grades) — with their source filenames and page numbers.

    Pass a clear, standalone question or topic, e.g.
    "Zara refund period", "travel insurance coverage for Class A", "annual leave Egypt".
    """
    return _tool_get_document_knowledge({"query": query}, user_id="mcp_claude_code")


@mcp.tool()
def kb_info() -> str:
    """Return basic info about the connected Azadea knowledge base collection."""
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          check_compatibility=False)
        total = qc.count(COLLECTION_NAME, exact=True).count
        return json.dumps({"collection": COLLECTION_NAME, "total_chunks": total,
                           "embedder": "OCI Cohere Embed v4 (dense) + BM25 (sparse)"})
    except Exception as e:
        return json.dumps({"collection": COLLECTION_NAME, "error": str(e)})


if __name__ == "__main__":
    mcp.run()
