"""Recover the client's Jul 6-7 ingestions into v2.

The client ingested via 8074 while it still pointed at the OLD collection, so
those (already-enriched) docs landed in docs_oci_ingested_azadea, not v2. This
copies the Jul 6-7 ADD/UPDATE docs verbatim OLD -> v2 (they're already enriched:
context_header + section_heading), atomic per-doc replace, plus version-sync
(retire an older revision in v2 if a newer one arrived). DELETE candidates are
only REPORTED (delete path logs no doc name), never auto-removed.
Read-only on OLD; writes only to v2.
"""
import json, os, re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from qdrant_client import QdrantClient, models as qm

qc = QdrantClient(url="http://localhost:6333", check_compatibility=False)
OLD, NEW = "docs_oci_ingested_azadea", "docs_oci_claude_v2"

def norm(s): return " ".join(str(s).replace("–","-").replace("—","-").split()).casefold()

def doc_chunks(coll, doc_id):
    out=[]; off=None
    while True:
        pts,off=qc.scroll(coll, limit=256,
            scroll_filter=qm.Filter(must=[qm.FieldCondition(key="doc_id",match=qm.MatchValue(value=doc_id))]),
            offset=off, with_payload=True, with_vectors=True)
        out+=pts
        if off is None: break
    return out

def all_docs(coll):
    s=set(); off=None
    while True:
        pts,off=qc.scroll(coll,limit=2000,offset=off,with_payload=["doc_id"],with_vectors=False)
        for p in pts:
            d=(p.payload or {}).get("doc_id")
            if d: s.add(d)
        if off is None: break
    return s

# base name = doc_id with trailing revision number stripped
VER=re.compile(r"^(.*[-–]\s*[A-Za-z0-9]{1,4})\s*[-–]\s*(\d{1,3})\s*$")
def base_ver(d):
    m=VER.match(norm(d)); return (m.group(1).strip(), int(m.group(2))) if m else (norm(d), None)

jul67=[l.strip() for l in open("/tmp/jul67_docs.txt") if l.strip()]
v2=all_docs(NEW); v2n={norm(d) for d in v2}

print(f"=== Recovering {len(jul67)} Jul 6-7 add/update docs into v2 ===")
copied=refreshed=retired=0
for d in jul67:
    src=doc_chunks(OLD, d)
    if not src:
        print(f"  SKIP (not in OLD): {d[:50]}"); continue
    existed = norm(d) in v2n
    # atomic per-doc replace in v2
    qc.delete(NEW, points_selector=qm.FilterSelector(filter=qm.Filter(
        must=[qm.FieldCondition(key="doc_id",match=qm.MatchValue(value=d))])))
    qc.upsert(NEW, points=[qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in src])
    copied += len(src); refreshed += existed
    # version-sync: retire older revisions of the same base name in v2
    b,v = base_ver(d)
    if v is not None:
        for od in list(v2):
            ob,ov = base_ver(od)
            if ob==b and ov is not None and ov < v and norm(od)!=norm(d):
                cnt=len(doc_chunks(NEW, od))
                qc.delete(NEW, points_selector=qm.FilterSelector(filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id",match=qm.MatchValue(value=od))])))
                retired += 1
                print(f"    retired older rev in v2: {od[:46]}  (superseded by -{v})")
    tag="refreshed" if existed else "ADDED"
    print(f"  {tag:9s} {d[:50]:52s} {len(src):3d} chunks")

print(f"\nsummary: {copied} chunks upserted | {refreshed} refreshed, {len(jul67)-refreshed} newly added | {retired} older revs retired")

# delete candidates: in v2 but NOT in current OLD, and NOT an intentionally-superseded rebuild version
sup={norm(x) for x in json.load(open('/tmp/v2_superseded_docids.json'))}
old=all_docs(OLD); oldn={norm(x) for x in old}
cand=[d for d in all_docs(NEW) if norm(d) not in oldn and norm(d) not in sup]
print(f"\n=== DELETE candidates (in v2, not in OLD, not a known-superseded rebuild doc): {len(cand)} ===")
print("  (reported only — NOT auto-deleted; review these)")
for d in sorted(cand)[:40]: print("   ?", d)
final=qc.count(NEW,exact=True).count
print(f"\nv2 now: {final} chunks, {len(all_docs(NEW))} docs")
