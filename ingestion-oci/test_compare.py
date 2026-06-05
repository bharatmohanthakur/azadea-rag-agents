#!/usr/bin/env python3
"""
Compare non-OCI (existing) vs OCI RAG API responses.

Tests both /query and /query/stream endpoints with the same queries,
comparing response quality, latency, and sources.

Usage:
    python test_compare.py
"""

import requests
import json
import time
import sys

# Endpoints
AZURE_URL = "http://localhost:7870"    # Existing rag_server_llm_chunked.py
OCI_URL = "http://localhost:7874"      # New rag_server_oci.py

TEST_QUERIES = [
    "What is the leave policy?",
    "How many days of annual leave do employees get?",
    "What is the meat handling procedure?",
    "What are the dress code requirements?",
    "How does the termination process work?",
]


def test_query(url: str, query: str, user_id: str = "test_compare") -> dict:
    """Test non-streaming /query endpoint."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{url}/query",
            json={"query": query, "user_id": user_id},
            timeout=30,
        )
        elapsed = time.time() - t0
        data = resp.json()
        return {
            "status": resp.status_code,
            "elapsed": round(elapsed, 3),
            "response": data.get("response", "")[:300],
            "route": data.get("metadata", {}).get("route", "unknown"),
            "sources": len(data.get("metadata", {}).get("sources", [])),
            "server_elapsed": data.get("metadata", {}).get("elapsed_sec", 0),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:200], "elapsed": time.time() - t0}


def test_stream(url: str, query: str, user_id: str = "test_compare_stream") -> dict:
    """Test SSE streaming /query/stream endpoint."""
    t0 = time.time()
    first_token = None
    tokens = []
    sources = []
    try:
        resp = requests.post(
            f"{url}/query/stream",
            json={"query": query, "user_id": user_id},
            stream=True,
            timeout=30,
        )
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if data.get("type") == "token":
                    if first_token is None:
                        first_token = time.time() - t0
                    tokens.append(data.get("text", ""))
                elif data.get("type") == "source_found":
                    sources.append(data.get("source", ""))
                elif data.get("type") == "done":
                    break

        total = time.time() - t0
        full_text = "".join(tokens)
        return {
            "status": 200,
            "ttft": round(first_token, 3) if first_token else None,
            "total": round(total, 3),
            "tokens": len(tokens),
            "sources": len(sources),
            "response": full_text[:300],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def main():
    print("=" * 80)
    print("RAG API Comparison: Azure vs OCI")
    print("=" * 80)

    # Check both servers are up
    for name, url in [("Azure", AZURE_URL), ("OCI", OCI_URL)]:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            data = resp.json()
            points = data.get("total_points", 0)
            print(f"  {name:6s} ({url}): {data.get('status', 'unknown')} — {points} points")
        except Exception as e:
            print(f"  {name:6s} ({url}): UNREACHABLE — {e}")
            if name == "OCI":
                print("\n  OCI server not running. Start with: python rag_server_oci.py")
                return

    print()
    print("-" * 80)

    # Non-streaming comparison
    print("\n1. NON-STREAMING /query COMPARISON\n")
    for query in TEST_QUERIES:
        print(f"  Query: {query}")
        azure = test_query(AZURE_URL, query, "azure_test")
        oci = test_query(OCI_URL, query, "oci_test")

        print(f"    Azure: {azure.get('elapsed', '?')}s | route={azure.get('route', '?')} | sources={azure.get('sources', 0)}")
        print(f"    OCI:   {oci.get('elapsed', '?')}s | route={oci.get('route', '?')} | sources={oci.get('sources', 0)}")
        print(f"    Azure response: {azure.get('response', '?')[:150]}...")
        print(f"    OCI   response: {oci.get('response', '?')[:150]}...")
        print()

    # Streaming comparison
    print("-" * 80)
    print("\n2. STREAMING /query/stream COMPARISON\n")
    for query in TEST_QUERIES[:3]:
        print(f"  Query: {query}")
        azure = test_stream(AZURE_URL, query, "azure_stream")
        oci = test_stream(OCI_URL, query, "oci_stream")

        print(f"    Azure: TTFT={azure.get('ttft', '?')}s | total={azure.get('total', '?')}s | tokens={azure.get('tokens', 0)}")
        print(f"    OCI:   TTFT={oci.get('ttft', '?')}s | total={oci.get('total', '?')}s | tokens={oci.get('tokens', 0)}")
        print(f"    Azure: {azure.get('response', '')[:150]}...")
        print(f"    OCI:   {oci.get('response', '')[:150]}...")
        print()

    print("=" * 80)
    print("COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
