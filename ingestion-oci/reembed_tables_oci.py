#!/usr/bin/env python3
"""
In-place re-embed of ALL table_summary chunks in the live OCI Qdrant collection
(docs_oci_ingested_azadea).

Bug being fixed: table chunks were embedded from a value-free structural summary
("Table with N rows and M columns. Header: ..."), so the actual row values
(brand names, countries, periods, amounts) were invisible to dense+sparse search.

Fix: re-embed each table chunk from  summary + full_table  (the row values), and
upsert in place — same point id, same payload, new dense+sparse vectors. Payload
untouched (so the LLM still sees full_table on retrieval exactly as before).
"""
import sys, time
sys.path.insert(0, ".")
from qdrant_client import QdrantClient, models as qm
from oci_pipeline import embed_dense_oci
from qdrant_utils import build_sparse_vectors

COLL = "docs_oci_ingested_azadea"
BATCH = 32                 # OCI Cohere embed input batch
TOKEN_GUARD = 6000         # fall back to summary-only above this (~4 chars/token)
qc = QdrantClient(url="http://localhost:6333", check_compatibility=False)


def tok(s):
    return max(1, len(s) // 4)


def embed_text_for(pl):
    summary = pl.get("text", "") or ""
    full = pl.get("full_table", "") or ""
    if not full:
        return summary                      # nothing to add
    combined = f"{summary}\n\n{full}"
    return combined if tok(combined) <= TOKEN_GUARD else summary


# 1) Pull every table_summary chunk (id + payload)
print(f"[1/3] scanning {COLL} for table_summary chunks ...", flush=True)
points = []
offset = None
while True:
    res, offset = qc.scroll(
        COLL,
        scroll_filter=qm.Filter(must=[
            qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value="table_summary"))
        ]),
        limit=512, offset=offset, with_payload=True, with_vectors=False,
    )
    points.extend(res)
    if offset is None:
        break
print(f"      found {len(points)} table_summary chunks", flush=True)

# 2) Re-embed in batches and upsert in place
print(f"[2/3] re-embedding in batches of {BATCH} ...", flush=True)
updated = skipped_empty = 0
t0 = time.time()
for i in range(0, len(points), BATCH):
    batch = points[i:i + BATCH]
    texts, metas = [], []
    for p in batch:
        pl = p.payload or {}
        if not (pl.get("full_table") or "").strip():
            skipped_empty += 1
            continue
        texts.append(embed_text_for(pl))
        metas.append(p)
    if not texts:
        continue
    dense = embed_dense_oci(texts)
    sparse = build_sparse_vectors(texts)
    upserts = [
        qm.PointStruct(id=p.id, payload=p.payload,
                       vector={"dense": d, "sparse": s})
        for p, d, s in zip(metas, dense, sparse)
    ]
    qc.upsert(COLL, points=upserts)
    updated += len(upserts)
    print(f"      {min(i + BATCH, len(points))}/{len(points)}  (updated={updated}, skipped_empty={skipped_empty})", flush=True)

dt = round(time.time() - t0, 1)
print(f"[3/3] done in {dt}s — updated {updated}, skipped {skipped_empty} (no full_table)", flush=True)
