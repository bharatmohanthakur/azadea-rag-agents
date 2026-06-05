"""
Oracle 26ai AI Vector Search — replaces Qdrant + fastembed BM25.

Single database for:
  - Dense vector search (VECTOR column + vector_distance)
  - Keyword search (Oracle Text CONTAINS)
  - Hybrid search (two queries + merge)
  - All metadata storage

Connection pool for multi-worker / high-concurrency.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import oracledb

logger = logging.getLogger("oracle_vectordb")

# Config
WALLET_DIR = os.getenv("ORACLE_WALLET_DIR", "/home/admincsp/graphiti_fixed_test/ingestion-oci/wallet")
# Credentials come ONLY from the environment (.env / secret manager) — never
# hardcoded in source. Fail loudly if a required secret is missing.
WALLET_PASSWORD = os.getenv("ORACLE_WALLET_PASSWORD")
DB_USER = os.getenv("ORACLE_DB_USER")
DB_PASSWORD = os.getenv("ORACLE_DB_PASSWORD")
if not all([WALLET_PASSWORD, DB_USER, DB_PASSWORD]):
    raise RuntimeError(
        "Missing Oracle credentials. Set ORACLE_WALLET_PASSWORD, ORACLE_DB_USER, "
        "ORACLE_DB_PASSWORD in the environment (.env)."
    )
DB_DSN = os.getenv("ORACLE_DB_DSN", "brainshiftdb_medium")
POOL_MIN = int(os.getenv("ORACLE_POOL_MIN", "2"))
POOL_MAX = int(os.getenv("ORACLE_POOL_MAX", "10"))
TABLE_NAME = os.getenv("ORACLE_TABLE_NAME", "rag_chunks")

_pool = None


def get_pool() -> oracledb.ConnectionPool:
    """Lazy connection pool — safe for multi-worker."""
    global _pool
    if _pool is None:
        _pool = oracledb.create_pool(
            user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN,
            config_dir=WALLET_DIR, wallet_location=WALLET_DIR,
            wallet_password=WALLET_PASSWORD,
            min=POOL_MIN, max=POOL_MAX, increment=1,
        )
        logger.info(f"Oracle pool created (min={POOL_MIN}, max={POOL_MAX})")
    return _pool


def _conn():
    """Acquire a connection from the pool."""
    return get_pool().acquire()


def _release(conn):
    """Release connection back to pool."""
    get_pool().release(conn)


# ---------------------------------------------------------------------------
# Table management
# ---------------------------------------------------------------------------

def ensure_table():
    """Create rag_chunks table + indexes if not exists."""
    conn = _conn()
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM user_tables WHERE table_name = UPPER(:1)", [TABLE_NAME])
        if cursor.fetchone()[0] > 0:
            logger.info(f"Table {TABLE_NAME} exists")
            return

        cursor.execute(f'''
            CREATE TABLE {TABLE_NAME} (
                chunk_id     VARCHAR2(64) PRIMARY KEY,
                doc_id       VARCHAR2(256) NOT NULL,
                source_file  VARCHAR2(512),
                chunk_type   VARCHAR2(50),
                text_content CLOB,
                full_table   CLOB,
                embedding    VECTOR(1536, FLOAT32),
                page         NUMBER,
                domain       VARCHAR2(100),
                func         VARCHAR2(100),
                variant      VARCHAR2(50),
                page_group   VARCHAR2(200),
                page_start   NUMBER,
                page_end     NUMBER,
                has_controls NUMBER(1) DEFAULT 0,
                has_notes    NUMBER(1) DEFAULT 0,
                has_tables   NUMBER(1) DEFAULT 0,
                figure_type  VARCHAR2(50),
                image_path   VARCHAR2(512),
                caption      VARCHAR2(1000),
                roles        VARCHAR2(1000),
                table_header VARCHAR2(1000),
                row_count    NUMBER,
                col_count    NUMBER,
                total_pages  NUMBER
            )
        ''')
        cursor.execute(f"CREATE INDEX {TABLE_NAME}_doc_idx ON {TABLE_NAME}(doc_id)")
        cursor.execute(f"CREATE INDEX {TABLE_NAME}_type_idx ON {TABLE_NAME}(chunk_type)")
        cursor.execute(f"""CREATE INDEX {TABLE_NAME}_text_idx ON {TABLE_NAME}(text_content)
            INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('SYNC(ON COMMIT)')""")
        try:
            cursor.execute(f"""CREATE VECTOR INDEX {TABLE_NAME}_vec_idx ON {TABLE_NAME}(embedding)
                ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE WITH TARGET ACCURACY 95""")
        except Exception as e:
            logger.warning(f"Vector index creation: {e}")
        conn.commit()
        logger.info(f"Created table {TABLE_NAME} with indexes")
    finally:
        _release(conn)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def hybrid_search(
    query_vec: List[float],
    query_text: str,
    top_k: int = 7,
    chunk_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid search: dense vector + Oracle Text keyword.
    Returns merged, deduplicated results.
    """
    conn = _conn()
    cursor = conn.cursor()
    try:
        vec_json = json.dumps(query_vec)
        keywords = _extract_keywords(query_text)
        results = {}

        # 1. Hybrid: keyword filter + vector rank (single query, ~200-500ms)
        #    Uses Oracle Text for filtering, HNSW for ranking within filtered set
        if keywords:
            try:
                # Wrap ORDER BY in a subquery + ROWNUM to prevent Oracle's optimizer
                # from silently using the HNSW index (which gives unstable top-k results
                # on this corpus regardless of APPROX keyword). Forces full distance scan
                # within the keyword-filtered set, then exact top-k.
                cursor.execute(f'''
                    SELECT chunk_id, doc_id, source_file, chunk_type, text_content,
                           full_table, page, text_score, dist
                    FROM (
                        SELECT chunk_id, doc_id, source_file, chunk_type, text_content,
                               full_table, page, SCORE(1) as text_score,
                               vector_distance(embedding, :1, COSINE) as dist
                        FROM {TABLE_NAME}
                        WHERE CONTAINS(text_content, :2, 1) > 0
                        ORDER BY dist
                    )
                    WHERE ROWNUM <= {top_k}
                ''', [vec_json, keywords])

                for row in cursor.fetchall():
                    cid = row[0]
                    results[cid] = {
                        "chunk_id": cid, "doc_id": row[1], "source_file": row[2],
                        "chunk_type": row[3], "text": _read_clob(row[4]),
                        "full_table": _read_clob(row[5]), "page": row[6],
                        "score": 1 - (row[8] or 0),
                        "search_type": "hybrid",
                    }
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}")

        # 2. Pure vector (HNSW, ~100-300ms) — catches semantic matches keywords missed
        try:
            cursor.execute(f'''
                SELECT chunk_id, doc_id, source_file, chunk_type, text_content,
                       full_table, page, distance
                FROM (
                    SELECT chunk_id, doc_id, source_file, chunk_type, text_content,
                           full_table, page, vector_distance(embedding, :1, COSINE) as distance
                    FROM {TABLE_NAME}
                    ORDER BY distance
                )
                WHERE ROWNUM <= {top_k}
            ''', [vec_json])

            for row in cursor.fetchall():
                cid = row[0]
                if cid not in results:
                    results[cid] = {
                        "chunk_id": cid, "doc_id": row[1], "source_file": row[2],
                        "chunk_type": row[3], "text": _read_clob(row[4]),
                        "full_table": _read_clob(row[5]), "page": row[6],
                        "score": 1 - (row[7] or 0),
                        "search_type": "vector",
                    }
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        sorted_results = sorted(results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]
    finally:
        _release(conn)


def scroll_tables(doc_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch table_summary chunks for a document (neighbor expansion)."""
    conn = _conn()
    cursor = conn.cursor()
    try:
        cursor.execute(f'''
            SELECT chunk_id, text_content, full_table, source_file, page
            FROM {TABLE_NAME}
            WHERE doc_id = :1 AND chunk_type = 'table_summary'
            FETCH FIRST :2 ROWS ONLY
        ''', [doc_id, limit])
        return [
            {"chunk_id": r[0], "text": _read_clob(r[1]), "full_table": _read_clob(r[2]),
             "source_file": r[3], "page": r[4]}
            for r in cursor.fetchall()
        ]
    finally:
        _release(conn)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def upsert_chunks(
    chunk_payloads: List[Dict[str, Any]],
    dense_vectors: List[List[float]],
) -> int:
    """Batch insert chunks with vectors. Skips duplicates."""
    conn = _conn()
    cursor = conn.cursor()
    cursor.setinputsizes(
        None, None, None, None,
        oracledb.DB_TYPE_CLOB, oracledb.DB_TYPE_CLOB,
        None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None
    )
    inserted = 0
    try:
        for pl, vec in zip(chunk_payloads, dense_vectors):
            pg = pl.get('page_group', '')
            if isinstance(pg, list): pg = json.dumps(pg)
            roles = pl.get('roles', '')
            if isinstance(roles, list): roles = ','.join(str(r) for r in roles)
            try:
                cursor.execute(f'''
                    INSERT INTO {TABLE_NAME} (chunk_id, doc_id, source_file, chunk_type,
                        text_content, full_table, embedding, page, domain, func, variant,
                        page_group, page_start, page_end, has_controls, has_notes, has_tables,
                        figure_type, image_path, caption, roles, table_header, row_count, col_count, total_pages)
                    VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15,:16,:17,:18,:19,:20,:21,:22,:23,:24,:25)
                ''', [
                    pl.get('chunk_id', '')[:64], pl.get('doc_id', '')[:256],
                    pl.get('source_file', '')[:512], pl.get('chunk_type', '')[:50],
                    pl.get('text', '') or '', pl.get('full_table', '') or '',
                    json.dumps(vec), pl.get('page'),
                    (pl.get('domain') or '')[:100] or None,
                    (pl.get('function') or '')[:100] or None,
                    (pl.get('variant') or '')[:50] or None,
                    str(pg)[:200] if pg else None,
                    pl.get('page_start'), pl.get('page_end'),
                    1 if pl.get('has_controls') else 0,
                    1 if pl.get('has_notes') else 0,
                    1 if pl.get('has_tables') else 0,
                    (pl.get('figure_type') or '')[:50] or None,
                    (pl.get('image_path') or '')[:512] or None,
                    (pl.get('caption') or '')[:1000] or None,
                    str(roles)[:1000] if roles else None,
                    (pl.get('header') or '')[:1000] or None,
                    pl.get('row_count'), pl.get('column_count'), pl.get('total_pages'),
                ])
                inserted += 1
            except oracledb.IntegrityError:
                pass
        conn.commit()
    finally:
        _release(conn)

    logger.info(f"Upserted {inserted} chunks to Oracle")
    return inserted


def delete_by_doc(doc_id: str) -> int:
    """Delete all chunks for a document."""
    conn = _conn()
    cursor = conn.cursor()
    try:
        cursor.execute(f"DELETE FROM {TABLE_NAME} WHERE doc_id = :1", [doc_id])
        deleted = cursor.rowcount
        conn.commit()
        return deleted
    finally:
        _release(conn)


def count_by_doc(doc_id: str) -> int:
    """Count chunks for a document."""
    conn = _conn()
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE doc_id = :1", [doc_id])
        return cursor.fetchone()[0]
    finally:
        _release(conn)


def get_health() -> Dict[str, Any]:
    """Health check: row count + table status."""
    conn = _conn()
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        total = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(DISTINCT doc_id) FROM {TABLE_NAME}")
        docs = cursor.fetchone()[0]
        return {
            "status": "ok",
            "backend": "oracle_26ai",
            "table": TABLE_NAME,
            "total_chunks": total,
            "total_docs": docs,
            "pool_busy": get_pool().busy,
            "pool_open": get_pool().opened,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        _release(conn)


def get_chunk_types(doc_id: str) -> Dict[str, int]:
    """Get chunk type breakdown for a document."""
    conn = _conn()
    cursor = conn.cursor()
    try:
        cursor.execute(f'''
            SELECT chunk_type, COUNT(*) FROM {TABLE_NAME}
            WHERE doc_id = :1 GROUP BY chunk_type
        ''', [doc_id])
        return {r[0]: r[1] for r in cursor.fetchall()}
    finally:
        _release(conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_clob(val) -> str:
    """Read CLOB value — may be string or LOB object."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return val.read()


def _extract_keywords(text: str) -> str:
    """Extract search keywords for Oracle Text CONTAINS query.
    Applies stemming ($) for better recall."""
    if not text:
        return ""
    # Remove special chars that break CONTAINS
    import re
    clean = re.sub(r'[^\w\s]', ' ', text)
    words = clean.split()
    # Take meaningful words (skip very short ones)
    keywords = [w for w in words if len(w) > 2][:8]
    if not keywords:
        return ""
    # Use OR for broader recall, $ for stemming
    return " OR ".join(f"${w}" for w in keywords)
