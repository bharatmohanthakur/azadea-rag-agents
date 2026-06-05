#!/usr/bin/env python3
"""
Parallel batch ingest ALL PDFs via the OCI ingestion service (port 8074).
Submits multiple docs concurrently, polls all in parallel.

Usage:
    nohup python batch_ingest_parallel.py > /tmp/oci_batch_ingest.log 2>&1 &

    # With custom concurrency:
    BATCH_SIZE=10 python batch_ingest_parallel.py
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVICE_URL = "http://localhost:8074"
PDF_ROOT = "/home/admincsp/multimodal-rag/azadea/data"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))  # Concurrent docs
POLL_INTERVAL = 3


def get_all_pdfs():
    pdfs = []
    for root, dirs, files in os.walk(PDF_ROOT):
        for f in files:
            if f.endswith(".pdf"):
                pdfs.append(f)
    return sorted(set(pdfs))


def get_ingested_docs():
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url="http://localhost:6333", check_compatibility=False)
        all_docs = set()
        offset = None
        while True:
            results, offset = qc.scroll(
                "docs_oci_ingested_azadea", limit=1000,
                with_payload=True, with_vectors=False,
                offset=offset,
            )
            for p in results:
                all_docs.add((p.payload or {}).get("doc_id", ""))
            if offset is None or not results:
                break
        return all_docs
    except Exception:
        return set()


def ingest_one(filename, idx, total):
    """Submit one PDF and poll until done. Returns (filename, status, chunks, elapsed)."""
    t0 = time.time()

    # Submit
    try:
        resp = requests.post(
            f"{SERVICE_URL}/document",
            json={"filename": filename, "operation": "add"},
            timeout=10,
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", "")[:100]
            return filename, "skip", 0, time.time() - t0, detail
        job_id = resp.json().get("job_id")
    except Exception as e:
        return filename, "error", 0, time.time() - t0, str(e)[:100]

    # Poll until done
    for _ in range(200):  # Max ~10 min
        time.sleep(POLL_INTERVAL)
        try:
            job = requests.get(f"{SERVICE_URL}/job/{job_id}", timeout=5).json()
            status = job.get("status")
            if status == "COMPLETED":
                chunks = job.get("chunks_created", 0)
                return filename, "ok", chunks, time.time() - t0, ""
            elif status == "FAILED":
                return filename, "fail", 0, time.time() - t0, job.get("error", "")[:100]
        except Exception:
            pass

    return filename, "timeout", 0, time.time() - t0, ""


def main():
    print("=" * 80)
    print(f"OCI Parallel Batch Ingestion (concurrency={BATCH_SIZE})")
    print("=" * 80)

    # Check service
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
    print(f"Estimated time: ~{len(remaining) * 30 // BATCH_SIZE // 60} minutes")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    stats = {"ok": 0, "fail": 0, "skip": 0, "error": 0, "timeout": 0, "total_chunks": 0}
    start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {}
        for i, pdf in enumerate(remaining, 1):
            future = executor.submit(ingest_one, pdf, i, len(remaining))
            futures[future] = (i, pdf)

        for future in as_completed(futures):
            idx, pdf = futures[future]
            completed += 1
            try:
                filename, result, chunks, elapsed, error = future.result()
                stats[result] = stats.get(result, 0) + 1
                stats["total_chunks"] += chunks

                short = filename[:55]
                if result == "ok":
                    print(f"[{completed:4d}/{len(remaining)}] OK    {short:55s} {chunks:3d} chunks  {elapsed:.0f}s")
                elif result == "fail":
                    print(f"[{completed:4d}/{len(remaining)}] FAIL  {short:55s} {error}")
                elif result == "skip":
                    print(f"[{completed:4d}/{len(remaining)}] SKIP  {short:55s} {error}")
                else:
                    print(f"[{completed:4d}/{len(remaining)}] {result:5s} {short:55s} {error}")

            except Exception as e:
                stats["error"] += 1
                print(f"[{completed:4d}/{len(remaining)}] ERR   {pdf[:55]} — {e}")

            # Progress every 20 docs
            if completed % 20 == 0:
                elapsed_total = time.time() - start
                rate = elapsed_total / completed
                eta = rate * (len(remaining) - completed)
                pts = requests.get(f"{SERVICE_URL}/health", timeout=5).json().get("total_points", "?")
                print(f"--- [{completed}/{len(remaining)}] OK={stats['ok']} Fail={stats['fail']} "
                      f"Skip={stats['skip']} | Chunks={stats['total_chunks']} | "
                      f"Points={pts} | {rate:.1f}s/doc | ETA={eta/60:.0f}min ---")

    elapsed = time.time() - start
    print("=" * 80)
    print(f"COMPLETE in {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"Stats: {json.dumps(stats)}")

    health = requests.get(f"{SERVICE_URL}/health", timeout=5).json()
    print(f"Collection: {health.get('total_points', 0)} total points")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
