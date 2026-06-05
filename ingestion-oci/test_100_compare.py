#!/usr/bin/env python3
"""
Compare 100 questions from logs: Azure vs OCI RAG APIs.
Saves detailed per-question report to /tmp/comparison_report.json
"""

import json
import re
import time
import requests
import sys

AZURE = "http://localhost:7870"
OCI = "http://localhost:7874"
QUESTIONS_FILE = "/tmp/test_100_questions.json"
REPORT_FILE = "/tmp/comparison_report.json"
TIMEOUT = 30


def strip_html(t):
    return re.sub(r'<[^>]+>', '', t).strip()


def query_api(url, query, user_id):
    t0 = time.time()
    try:
        r = requests.post(f"{url}/query", json={"query": query, "user_id": user_id}, timeout=TIMEOUT)
        elapsed = time.time() - t0
        d = r.json()
        return {
            "status": "ok",
            "elapsed": round(elapsed, 2),
            "route": d.get("metadata", {}).get("route", "?"),
            "sources": len(d.get("metadata", {}).get("sources", [])),
            "response": strip_html(d.get("response", "")),
            "raw_response": d.get("response", "")[:500],
        }
    except requests.exceptions.Timeout:
        return {"status": "timeout", "elapsed": round(time.time() - t0, 2), "response": ""}
    except Exception as e:
        return {"status": "error", "elapsed": round(time.time() - t0, 2), "error": str(e)[:100], "response": ""}


def main():
    questions = json.loads(open(QUESTIONS_FILE).read())
    print(f"Testing {len(questions)} questions against Azure ({AZURE}) and OCI ({OCI})")

    # Verify servers
    for name, url in [("Azure", AZURE), ("OCI", OCI)]:
        try:
            h = requests.get(f"{url}/health", timeout=5).json()
            print(f"  {name}: {h.get('status')} — {h.get('total_points', 0)} points")
        except:
            print(f"  {name}: UNREACHABLE")
            return

    report = []
    azure_wins = oci_wins = ties = errors = 0
    azure_total_time = oci_total_time = 0

    for i, q in enumerate(questions, 1):
        sys.stdout.write(f"\r[{i:3d}/{len(questions)}] ")
        sys.stdout.flush()

        azure_r = query_api(AZURE, q, f"azure_test_{i}")
        oci_r = query_api(OCI, q, f"oci_test_{i}")

        azure_total_time += azure_r.get("elapsed", 0)
        oci_total_time += oci_r.get("elapsed", 0)

        # Simple quality heuristics
        az_ok = azure_r["status"] == "ok" and len(azure_r["response"]) > 20
        oci_ok = oci_r["status"] == "ok" and len(oci_r["response"]) > 20

        if az_ok and not oci_ok:
            winner = "azure"
            azure_wins += 1
        elif oci_ok and not az_ok:
            winner = "oci"
            oci_wins += 1
        elif not az_ok and not oci_ok:
            winner = "neither"
            errors += 1
        else:
            # Both answered — compare response length and quality
            az_len = len(azure_r["response"])
            oci_len = len(oci_r["response"])

            # Check if one is a clarification and other is a direct answer
            az_is_clarify = any(kw in azure_r["response"].lower() for kw in ["are you asking about", "which country", "please specify", "could you clarify"])
            oci_is_clarify = any(kw in oci_r["response"].lower() for kw in ["are you asking about", "which country", "please specify", "could you clarify"])

            if az_is_clarify == oci_is_clarify:
                winner = "tie"
                ties += 1
            elif oci_is_clarify and not az_is_clarify:
                # OCI asked clarification = smarter
                winner = "oci"
                oci_wins += 1
            else:
                winner = "azure"
                azure_wins += 1

        entry = {
            "index": i,
            "question": q,
            "winner": winner,
            "azure": {
                "status": azure_r["status"],
                "elapsed": azure_r.get("elapsed"),
                "route": azure_r.get("route"),
                "sources": azure_r.get("sources"),
                "response_length": len(azure_r.get("response", "")),
                "response": azure_r.get("response", "")[:500],
            },
            "oci": {
                "status": oci_r["status"],
                "elapsed": oci_r.get("elapsed"),
                "route": oci_r.get("route"),
                "sources": oci_r.get("sources"),
                "response_length": len(oci_r.get("response", "")),
                "response": oci_r.get("response", "")[:500],
            },
        }
        report.append(entry)

    # Save full report
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    total = len(questions)
    print(f"\n\n{'=' * 80}")
    print(f"COMPARISON REPORT: {total} Questions")
    print(f"{'=' * 80}")
    print(f"Azure wins:  {azure_wins:3d} ({azure_wins*100//total}%)")
    print(f"OCI wins:    {oci_wins:3d} ({oci_wins*100//total}%)")
    print(f"Ties:        {ties:3d} ({ties*100//total}%)")
    print(f"Both failed: {errors:3d} ({errors*100//total}%)")
    print(f"")
    print(f"Avg latency: Azure={azure_total_time/total:.2f}s  OCI={oci_total_time/total:.2f}s")
    print(f"")
    print(f"Report: {REPORT_FILE}")

    # Print each question result
    print(f"\n{'=' * 80}")
    print(f"DETAILED RESULTS")
    print(f"{'=' * 80}")
    for e in report:
        q = e["question"][:70]
        w = e["winner"]
        az_t = e["azure"]["elapsed"]
        oci_t = e["oci"]["elapsed"]
        az_len = e["azure"]["response_length"]
        oci_len = e["oci"]["response_length"]
        marker = "<<<" if w == "oci" else ">>>" if w == "azure" else "==="
        print(f"Q{e['index']:3d} [{w:6s}] {marker} Az={az_t:.1f}s/{az_len:4d}ch  OCI={oci_t:.1f}s/{oci_len:4d}ch  {q}")


if __name__ == "__main__":
    main()
