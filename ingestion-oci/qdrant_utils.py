"""
Qdrant utilities — sparse vectors, collection management, upsert.
Extracted from azure_doc_intelligence_qdrant.py with NO Azure dependencies.
"""

import os
import uuid
import logging
from typing import Any, Dict, List

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models as qm
from qdrant_client.http import models as qmodels

logger = logging.getLogger("qdrant_utils")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DENSE_NAME = "dense"
SPARSE_NAME = "sparse"

# ---------------------------------------------------------------------------
# Sparse (BM25 via fastembed — fully local, no cloud API)
# ---------------------------------------------------------------------------
SPARSE_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")


def _to_sparse_vector(sv) -> qmodels.SparseVector:
    """Normalize fastembed sparse output into Qdrant SparseVector."""
    if hasattr(sv, "indices") and hasattr(sv, "values"):
        return qmodels.SparseVector(indices=list(sv.indices), values=list(sv.values))
    if isinstance(sv, dict) and "indices" in sv and "values" in sv:
        return qmodels.SparseVector(indices=list(sv["indices"]), values=list(sv["values"]))
    if isinstance(sv, (tuple, list)) and len(sv) == 2:
        return qmodels.SparseVector(indices=list(sv[0]), values=list(sv[1]))
    for cand in ("to_dict", "_asdict"):
        if hasattr(sv, cand):
            d = getattr(sv, cand)()
            if isinstance(d, dict) and "indices" in d and "values" in d:
                return qmodels.SparseVector(indices=list(d["indices"]), values=list(d["values"]))
    raise TypeError(f"Unsupported sparse embedding type: {type(sv)}")


def build_sparse_vectors(texts: List[str]) -> List[qmodels.SparseVector]:
    """Convert texts → Qdrant SparseVector using fastembed BM25."""
    return [_to_sparse_vector(sv) for sv in SPARSE_MODEL.embed(texts)]


def build_sparse_query_vector(query_text: str) -> qmodels.SparseVector:
    sv = next(SPARSE_MODEL.embed([query_text]))
    return _to_sparse_vector(sv)


# ---------------------------------------------------------------------------
# Qdrant collection management
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient, name: str, vector_dim: int):
    """Create collection with named dense + sparse (BM25/IDF) vectors."""
    try:
        client.get_collection(name)
        return  # Already exists
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_NAME: qm.VectorParams(size=vector_dim, distance=qm.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_NAME: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF),
        },
    )
    client.update_collection(
        collection_name=name,
        optimizer_config=qm.OptimizersConfigDiff(
            indexing_threshold=10_000,
            default_segment_number=2,
        ),
    )
    logger.info(f"Created collection '{name}' (dense={vector_dim}, sparse=IDF)")


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_typed_chunks(
    client: QdrantClient,
    collection: str,
    chunk_payloads: List[Dict[str, Any]],
    dense_vectors: List[List[float]],
    sparse_vectors: List[qmodels.SparseVector],
):
    """Upsert typed chunks with full payload metadata + both vector types."""
    assert len(chunk_payloads) == len(dense_vectors) == len(sparse_vectors)
    points = []
    for payload, dvec, svec in zip(chunk_payloads, dense_vectors, sparse_vectors):
        pid = payload.get("chunk_id", uuid.uuid4().hex)
        points.append(
            qm.PointStruct(
                id=pid,
                payload=payload,
                vector={DENSE_NAME: dvec, SPARSE_NAME: svec},
            )
        )
    client.upsert(collection_name=collection, points=points)

    type_counts: Dict[str, int] = {}
    for p in chunk_payloads:
        ct = p.get("chunk_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    logger.info(f"Upserted {len(points)} typed chunks → {collection} ({summary})")
    return len(points)


# ---------------------------------------------------------------------------
# Per-document delete / count (mirror of oracle_vectordb, for the live Qdrant store)
# ---------------------------------------------------------------------------

def count_by_doc(client: QdrantClient, collection: str, doc_id: str) -> int:
    """Count chunks belonging to a document."""
    try:
        return client.count(
            collection_name=collection,
            count_filter=qm.Filter(must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
            ]),
            exact=True,
        ).count
    except Exception:
        return 0


def delete_by_doc(client: QdrantClient, collection: str, doc_id: str) -> int:
    """Delete all chunks for a document. Returns the count removed."""
    n = count_by_doc(client, collection, doc_id)
    if n:
        client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
            ])),
        )
    return n
