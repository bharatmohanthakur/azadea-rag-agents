#!/usr/bin/env python3
"""
Batch ingest ALL PDFs via the OCI ingestion service (port 8074).
Runs sequentially, logs progress, skips already-ingested docs.

Usage:
    nohup python batch_ingest.py > /tmp/oci_batch_ingest.log 2>&1 &
"""

import os
import sys
import time
import json
import requests
from pathlib import Path

SERVICE_URL = "http://localhost:8074"
PDF_ROOT = "/home/admincsp/multimodal-rag/azadea/data"
MAX_RETRIES = 2
POLL_INTERVAL = 5  # seconds


def get_all_pdfs():
    pdfs = []
    for root, dirs, files in os.walk(PDF_ROOT):
        for f in files:
            if f.endswith(".pdf"):
                pdfs.append(f)
    return sorted(set(pdfs))  # Deduplicate by filename


def get_ingested_docs():
    """Get doc_ids already in the OCI collection."""
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url="http://localhost:6333", check_compatibility=False)
        results, _ = qc.scroll(
            "docs_oci_ingested_azadea", limit=5000,
            with_payload=True, with_vectors=False,
        )
        return set((p.payload or {}).get("doc_id", "") for p in results)
    except Exception:
        return set()


def submit_and_wait(filename, idx, total):
    """Submit one PDF and poll until done."""
    # Submit
    try:
        resp = requests.post(
            f"{SERVICE_URL}/document",
            json={"filename": filename, "operation": "add"},
            timeout=10,
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text[:200])
            print(f"[{idx}/{total}] SKIP {filename[:60]} — {detail}")
            return "skipped", 0
        job_id = resp.json().get("job_id")
    except Exception as e:
        print(f"[{idx}/{total}] ERROR submit {filename[:60]} — {e}")
        return "error", 0

    # Poll
    for _ in range(120):  # Max 10 minutes per doc
        time.sleep(POLL_INTERVAL)
        try:
            job = requests.get(f"{SERVICE_URL}/job/{job_id}", timeout=5).json()
            status = job.get("status", "UNKNOWN")
            if status == "COMPLETED":
                chunks = job.get("chunks_created", 0)
                print(f"[{idx}/{total}] OK    {filename[:60]} — {chunks} chunks")
                return "ok", chunks
            elif status == "FAILED":
                error = job.get("error", "")[:100]
                print(f"[{idx}/{total}] FAIL  {filename[:60]} — {error}")
                return "failed", 0
        except Exception:
            pass

    print(f"[{idx}/{total}] TIMEOUT {filename[:60]}")
    return "timeout", 0


def main():
    print("=" * 80)
    print("OCI Batch Ingestion")
    print("=" * 80)

    # Check service is up
    try:
        health = requests.get(f"{SERVICE_URL}/health", timeout=5).json()
        print(f"Service: {health.get('status')} — {health.get('total_points', 0)} existing points")
    except Exception as e:
        print(f"Service not reachable: {e}")
        sys.exit(1)

    all_pdfs = get_all_pdfs()
    ingested = get_ingested_docs()
    remaining = [f for f in all_pdfs if os.path.splitext(f)[0] not in ingested]

    print(f"Total PDFs: {len(all_pdfs)}")
    print(f"Already ingested: {len(ingested)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    stats = {"ok": 0, "failed": 0, "skipped": 0, "timeout": 0, "error": 0, "total_chunks": 0}
    start = time.time()

    for i, pdf in enumerate(remaining, 1):
        result, chunks = submit_and_wait(pdf, i, len(remaining))
        stats[result] = stats.get(result, 0) + 1
        stats["total_chunks"] += chunks

        # Progress every 10 docs
        if i % 10 == 0:
            elapsed = time.time() - start
            rate = elapsed / i
            eta = rate * (len(remaining) - i)
            print(f"--- Progress: {i}/{len(remaining)} | "
                  f"OK={stats['ok']} Failed={stats['failed']} Skip={stats['skipped']} | "
                  f"Chunks={stats['total_chunks']} | "
                  f"Rate={rate:.1f}s/doc | ETA={eta/60:.0f}min ---")

    elapsed = time.time() - start
    print("=" * 80)
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Results: {json.dumps(stats)}")

    # Final health
    health = requests.get(f"{SERVICE_URL}/health", timeout=5).json()
    print(f"Collection: {health.get('total_points', 0)} total points")


if __name__ == "__main__":
    main()
