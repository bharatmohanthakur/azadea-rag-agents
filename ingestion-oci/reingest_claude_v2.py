"""
Blue/green corpus re-ingest into a SEPARATE Qdrant collection.

Rebuilds every disk PDF through the full Claude pipeline (Claude DU → Claude
grouping → section chunks → context headers → Cohere embed) into
`docs_oci_claude_v2` — the live collection `docs_oci_ingested_azadea` is never
touched. Markdown artifacts go to their own directory too.

After the PDFs, chunks for index-only documents (docs that exist in the old
collection but have no source PDF on disk — e.g. client-pushed docs like the
ABS Charter) are COPIED verbatim from the old collection, so the new collection
is a superset, never missing anything the old one had.

Resumable: docs already present in the target collection are skipped.
Run:  OCI env + ANTHROPIC key come from the root .env (loaded below).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

NEW_COLLECTION = os.environ.setdefault("OCI_QDRANT_COLLECTION", "docs_oci_claude_v2")
os.environ.setdefault("DU_BACKEND", "claude")
os.environ.setdefault("GROUPING_BACKEND", "claude")
os.environ.setdefault("SECTION_CHUNKING", "1")
os.environ.setdefault("CONTEXT_HEADERS", "1")

OLD_COLLECTION = "docs_oci_ingested_azadea"
DATA_ROOT = Path("/home/admincsp/multimodal-rag/azadea/data/data")
MD_OUT = Path("/home/admincsp/multimodal-rag/azadea/md_out_data_oci_claude_v2")
CONCURRENCY = int(os.getenv("REINGEST_CONCURRENCY", "10"))
PROGRESS = Path("/tmp/reingest_claude_v2.progress.jsonl")

# import AFTER env is set — ingest_pipeline reads OCI_QDRANT_COLLECTION at import
import ingest_pipeline as ip            # noqa: E402
import qdrant_utils                      # noqa: E402

qc = ip._qdrant_client


def log(rec):
    rec["ts"] = time.strftime("%H:%M:%S")
    with open(PROGRESS, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(rec, flush=True)


def indexed_docs(coll):
    s = set(); off = None
    while True:
        pts, off = qc.scroll(coll, limit=1000, offset=off,
                             with_payload=["doc_id"], with_vectors=False)
        for p in pts:
            s.add((p.payload or {}).get("doc_id"))
        if off is None:
            break
    return s


def process_one(pdf: Path):
    doc_id = pdf.stem
    try:
        md_path, _ = ip.process_pdf_oci(pdf, MD_OUT)
        n = ip.ingest_md_oci(md_path)
        return {"doc": doc_id, "ok": True, "chunks": n}
    except Exception as e:
        return {"doc": doc_id, "ok": False, "error": str(e)[:200]}


def _norm(s):
    return " ".join(str(s).replace("–", "-").replace("—", "-").split()).casefold()


def load_superseded():
    """Version-sync kill-list built from the live collection: doc_ids whose base
    name has a NEWER revision (e.g. 'F&A Costing - G - 1' superseded by 'G - 2'
    pushed by the client). These must never enter v2 — not rebuilt from disk,
    not copied from the old collection."""
    p = Path("/tmp/v2_superseded_docids.json")
    if not p.exists():
        return set()
    return {_norm(d) for d in json.load(open(p))}


def main():
    MD_OUT.mkdir(parents=True, exist_ok=True)
    qdrant_utils.ensure_collection(qc, NEW_COLLECTION, vector_dim=1536)
    superseded = load_superseded()

    # SYNC FIRST: purge any superseded doc_ids already ingested into v2 by the
    # earlier (pre-sync) run, so the collection never carries version conflicts.
    purged = 0
    for d in list(indexed_docs(NEW_COLLECTION)):
        if d and _norm(d) in superseded:
            qdrant_utils.delete_by_doc(qc, NEW_COLLECTION, d)
            purged += 1
    log({"phase": "sync", "superseded_total": len(superseded),
         "purged_from_v2": purged})

    pdfs = sorted(DATA_ROOT.rglob("*.pdf"))
    done = indexed_docs(NEW_COLLECTION)
    skipped_old = [p for p in pdfs if _norm(p.stem) in superseded]
    todo = [p for p in pdfs
            if p.stem not in done and _norm(p.stem) not in superseded]
    log({"phase": "start", "pdfs_on_disk": len(pdfs),
         "already_done": len(done), "skipped_superseded": len(skipped_old),
         "todo": len(todo),
         "collection": NEW_COLLECTION, "concurrency": CONCURRENCY})

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(process_one, p): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            rec["n"] = f"{i}/{len(todo)}"
            ok += rec["ok"]; fail += (not rec["ok"])
            log(rec)

    log({"phase": "pdfs_done", "ok": ok, "failed": fail,
         "mins": round((time.time() - t0) / 60, 1)})

    # ── copy index-only docs (no source PDF on disk) from old collection ──
    new_docs = indexed_docs(NEW_COLLECTION)
    old_docs = indexed_docs(OLD_COLLECTION)
    disk_stems = {p.stem for p in pdfs}
    orphans = sorted(d for d in old_docs
                     if d not in new_docs and d not in disk_stems and d
                     and _norm(d) not in superseded)
    log({"phase": "copy_orphans", "count": len(orphans)})
    copied = 0
    from qdrant_client import models as qm
    for d in orphans:
        off = None
        while True:
            pts, off = qc.scroll(
                OLD_COLLECTION, limit=200, offset=off,
                scroll_filter=qm.Filter(must=[qm.FieldCondition(
                    key="doc_id", match=qm.MatchValue(value=d))]),
                with_payload=True, with_vectors=True)
            if pts:
                qc.upsert(NEW_COLLECTION, points=[
                    qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                    for p in pts])
                copied += len(pts)
            if off is None:
                break
    log({"phase": "orphans_copied", "chunks": copied, "docs": len(orphans)})

    final = qc.count(NEW_COLLECTION, exact=True).count
    log({"phase": "DONE", "new_collection_chunks": final,
         "total_mins": round((time.time() - t0) / 60, 1)})


if __name__ == "__main__":
    main()
