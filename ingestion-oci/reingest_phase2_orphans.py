"""
Phase 2 of the blue/green rebuild: re-process the 169 'client-only' documents
from their real source files (discovered at the PDF_ROOT top level — 119 PDF +
50 DOCX) through the full enriched Claude pipeline, into docs_oci_claude_v2.

Each ingest atomically replaces the verbatim copy phase-1 made for that doc.
If the old doc_id (from the live collection) differs cosmetically from the
source file's stem (dash/whitespace variants), the stale-named copy is removed
so v2 holds exactly one doc_id per document.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

os.environ.setdefault("OCI_QDRANT_COLLECTION", "docs_oci_claude_v2")
os.environ.setdefault("DU_BACKEND", "claude")
os.environ.setdefault("GROUPING_BACKEND", "claude")
os.environ.setdefault("SECTION_CHUNKING", "1")
os.environ.setdefault("CONTEXT_HEADERS", "1")

MD_OUT = Path("/home/admincsp/multimodal-rag/azadea/md_out_data_oci_claude_v2")
CONCURRENCY = int(os.getenv("REINGEST_CONCURRENCY", "10"))
PROGRESS = Path("/tmp/reingest_phase2.progress.jsonl")

import ingest_pipeline as ip                      # noqa: E402
import qdrant_utils                                # noqa: E402
from simple_converters import docx_to_markdown, text_to_markdown  # noqa: E402

qc = ip._qdrant_client
NEW = os.environ["OCI_QDRANT_COLLECTION"]


def log(rec):
    rec["ts"] = time.strftime("%H:%M:%S")
    with open(PROGRESS, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(rec, flush=True)


def process_one(old_doc_id: str, src: Path):
    try:
        ext = src.suffix.lower()
        if ext == ".pdf":
            md_path, _ = ip.process_pdf_oci(src, MD_OUT)
        elif ext == ".docx":
            md_path = docx_to_markdown(src, MD_OUT)
        else:
            md_path = text_to_markdown(src, MD_OUT)
        n = ip.ingest_md_oci(md_path)
        # remove the phase-1 verbatim copy if it lives under a variant doc_id
        if old_doc_id and old_doc_id != md_path.stem:
            removed = qdrant_utils.delete_by_doc(qc, NEW, old_doc_id)
            if removed:
                log({"doc": old_doc_id, "note": f"removed stale-named copy ({removed} chunks)"})
        return {"doc": md_path.stem, "ok": True, "chunks": n, "src_type": ext}
    except Exception as e:
        return {"doc": old_doc_id, "ok": False, "error": str(e)[:200]}


def main():
    MD_OUT.mkdir(parents=True, exist_ok=True)
    sources = json.load(open("/tmp/v2_orphan_sources.json"))
    items = [(doc, Path(p)) for doc, p in sources.items() if Path(p).exists()]
    log({"phase": "phase2_start", "orphans": len(sources),
         "with_source": len(items), "concurrency": CONCURRENCY})

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(process_one, d, p): d for d, p in items}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            rec["n"] = f"{i}/{len(items)}"
            ok += rec.get("ok", False); fail += (not rec.get("ok", False))
            log(rec)

    final = qc.count(NEW, exact=True).count
    log({"phase": "PHASE2_DONE", "ok": ok, "failed": fail,
         "v2_chunks": final, "mins": round((time.time() - t0) / 60, 1)})


if __name__ == "__main__":
    main()
