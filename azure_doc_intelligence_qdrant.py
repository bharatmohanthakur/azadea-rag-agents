#!/usr/bin/env python3
"""
Markdown → SEMANTIC chunks → Qdrant hybrid (dense + sparse)

Semantic chunking: Uses embedding similarity to detect topic boundaries
Dense: Azure OpenAI embeddings (e.g., text-embedding-3-large)
Sparse: BM25 via fastembed → normalized to Qdrant SparseVector
Hybrid search: server-side RRF fusion with Prefetch (dense + sparse)

No argparse; edit CONFIG below or set env vars.
"""

import os
import re
import glob
import uuid
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# =========================
# CONFIG (EDIT THESE)
# =========================
INPUT_MD_GLOB   = "./md_out/*.md"          # Markdown files to ingest
COLLECTION_NAME = "docs_hybrid_azure_azadea"      # Qdrant collection
MAX_TOKENS      = 1000                     # Reduced for semantic chunks
OVERLAP_TOKENS  = 150                      # Reduced overlap
EMBED_BATCH     = 64
EMBED_TIMEOUT_S = 60.0
SIMILARITY_THRESHOLD = 0.75                # Break chunk when similarity drops below this
MIN_CHUNK_SIZE  = 100                      # Minimum tokens per chunk

# If RUN_QUERY is None → ingestion. If a string → run a hybrid search for that query.
RUN_QUERY: Optional[str] = None
TOP_K = 10

# =========================
# Azure OpenAI (dense)
# =========================
from openai import AzureOpenAI

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")  or os.getenv("OPENAI_API_KEY")
AZURE_EMBED_DEPLOY    = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in env.")

aoai = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01",
)

def embed_dense_azure(texts: List[str]) -> List[List[float]]:
    """Batch embeddings from Azure OpenAI (dense). Retries with backoff."""
    vectors: List[List[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        for attempt in range(5):
            try:
                resp = aoai.embeddings.create(
                    input=batch,
                    model=AZURE_EMBED_DEPLOY,
                    timeout=EMBED_TIMEOUT_S,
                )
                vectors.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                sleep = min(2 ** attempt, 16)
                print(f"[azure-embed] retry {attempt+1}/5 after error: {e} (sleep {sleep}s)")
                time.sleep(sleep)
        else:
            raise RuntimeError("Azure embeddings failed after retries.")
    return vectors

def infer_embedding_dim() -> int:
    return len(embed_dense_azure(["probe"])[0])

# =========================
# Sparse (BM25 via fastembed)
# =========================
from fastembed import SparseTextEmbedding
from qdrant_client.http import models as qmodels

SPARSE_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")  # BM25-style sparse

def _to_sparse_vector(sv) -> qmodels.SparseVector:
    """
    Normalize fastembed sparse output into Qdrant SparseVector.
    Handles:
      - object with .indices / .values (SparseEmbedding)
      - dict {"indices": [...], "values": [...]}
      - tuple/list (indices, values)
      - objects with .to_dict() / ._asdict()
    """
    # Object with attributes (newer fastembed: SparseEmbedding)
    if hasattr(sv, "indices") and hasattr(sv, "values"):
        return qmodels.SparseVector(indices=list(sv.indices), values=list(sv.values))

    # Dict form
    if isinstance(sv, dict) and "indices" in sv and "values" in sv:
        return qmodels.SparseVector(indices=list(sv["indices"]), values=list(sv["values"]))

    # Tuple/list form
    if isinstance(sv, (tuple, list)) and len(sv) == 2:
        indices, values = sv
        return qmodels.SparseVector(indices=list(indices), values=list(values))

    # Alt method to dict
    for cand in ("to_dict", "_asdict"):
        if hasattr(sv, cand):
            d = getattr(sv, cand)()
            if isinstance(d, dict) and "indices" in d and "values" in d:
                return qmodels.SparseVector(indices=list(d["indices"]), values=list(d["values"]))

    raise TypeError(f"Unsupported sparse embedding type: {type(sv)}")

def build_sparse_vectors(texts: List[str]) -> List[qmodels.SparseVector]:
    """Convert texts → Qdrant SparseVector(indices, values) using fastembed BM25."""
    vecs: List[qmodels.SparseVector] = []
    for sv in SPARSE_MODEL.embed(texts):
        vecs.append(_to_sparse_vector(sv))
    return vecs

def build_sparse_query_vector(query_text: str) -> qmodels.SparseVector:
    sv = next(SPARSE_MODEL.embed([query_text]))
    return _to_sparse_vector(sv)

# =========================
# Qdrant (hybrid setup)
# =========================
from qdrant_client import QdrantClient, models as qm

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"

def ensure_collection(client: QdrantClient, name: str, vector_dim: int):
    """Create collection with named dense vector + sparse (BM25/IDF)."""
    exists = False
    try:
        client.get_collection(name)
        exists = True
    except Exception:
        exists = False

    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_NAME: qm.VectorParams(size=vector_dim, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_NAME: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
            },
        )
        client.update_collection(
            collection_name=name,
            optimizer_config=qm.OptimizersConfigDiff(
                indexing_threshold=10_000,
                default_segment_number=2,
            )
        )
        print(f"[qdrant] created '{name}' (dense={vector_dim}, sparse=IDF)")

def upsert_hybrid_points(
    client: QdrantClient,
    collection: str,
    chunks: List[str],
    meta_base: Dict[str, Any],
    dense_vectors: List[List[float]],
    sparse_vectors: List[qmodels.SparseVector],
):
    """Upsert points containing BOTH dense and sparse vectors."""
    assert len(chunks) == len(dense_vectors) == len(sparse_vectors)
    points: List[qm.PointStruct] = []
    for idx, (text, dvec, svec) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        pid = uuid.uuid4().hex
        payload = dict(meta_base)
        payload.update({"chunk_index": idx, "text": text})
        points.append(
            qm.PointStruct(
                id=pid,
                payload=payload,
                vector={
                    DENSE_NAME: dvec,  # list[float]
                    SPARSE_NAME: svec, # SparseVector(indices, values)
                },
            )
        )
    client.upsert(collection_name=collection, points=points)
    print(f"[qdrant] upserted {len(points)} points → {collection}")

def hybrid_search(
    client: QdrantClient,
    collection: str,
    query_text: str,
    top_k: int = 10,
    prefetch_k: int = 50,
    where: Optional[qm.Filter] = None,
) -> List[Dict[str, Any]]:
    """Dense + sparse prefetch, server-side RRF fusion."""
    dense_q  = embed_dense_azure([query_text])[0]
    sparse_q = build_sparse_query_vector(query_text)

    res = client.query_points(
        collection_name=collection,
        prefetch=[
            qm.Prefetch(query=dense_q,  using=DENSE_NAME,  limit=prefetch_k, filter=where),
            qm.Prefetch(query=sparse_q, using=SPARSE_NAME, limit=prefetch_k, filter=where),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=top_k,
    )
    hits: List[Dict[str, Any]] = []
    for p in res.points:
        pl = p.payload or {}
        txt = (pl.get("text") or "")
        hits.append({
            "id": p.id,
            "score": p.score,
            "source_file": pl.get("source_file"),
            "doc_id": pl.get("doc_id"),
            "chunk_index": pl.get("chunk_index"),
            "preview": txt[:300] + ("…" if len(txt) > 300 else ""),
        })
    return hits

# =========================
# Semantic Chunking
# =========================
def _count_tokens(text: str) -> int:
    try:
        import tiktoken  # lazy import
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def split_into_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences with character positions.
    Returns: List of (sentence, start_pos, end_pos) tuples
    """
    # Sentence pattern: ends with . ! ? followed by space or newline
    sentence_pattern = re.compile(r'([.!?]+)(?:\s+|\n+|$)')
    
    sentences = []
    start_pos = 0
    
    for match in sentence_pattern.finditer(text):
        end_pos = match.end()
        sentence = text[start_pos:end_pos].strip()
        
        if sentence and len(sentence) > 20:  # Skip very short sentences
            sentences.append((sentence, start_pos, end_pos))
        
        start_pos = end_pos
    
    # Add the last sentence if it doesn't end with punctuation
    if start_pos < len(text):
        sentence = text[start_pos:].strip()
        if sentence and len(sentence) > 20:
            sentences.append((sentence, start_pos, len(text)))
    
    return sentences


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def find_semantic_boundaries(embeddings: List[List[float]], threshold: float = SIMILARITY_THRESHOLD) -> List[int]:
    """
    Find indices where semantic boundaries occur (similarity drops below threshold).
    Returns list of sentence indices where a new chunk should start.
    """
    boundaries = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity(embeddings[i], embeddings[i + 1])
        if similarity < threshold:
            boundaries.append(i + 1)
    return boundaries


def extract_markdown_tables(text: str) -> List[tuple]:
    """
    Extract markdown tables from text as atomic units.
    Returns: List of (table_text, start_pos, end_pos) tuples
    """
    tables = []
    # Pattern to match markdown tables (lines starting with | or spaces+|)
    table_pattern = re.compile(
        r'(?:^|\n)((?:\s*\|[^\n]+\|\s*\n)+)',
        re.MULTILINE
    )
    
    for match in table_pattern.finditer(text):
        table_text = match.group(1).strip()
        # Ensure it's a valid table (has at least 2 rows and header separator)
        lines = table_text.split('\n')
        if len(lines) >= 2:
            # Check for header separator (---|---|---)
            has_separator = any('---' in line or '| --- |' in line.replace(' ', '') for line in lines[:3])
            if has_separator or len(lines) >= 3:  # Valid table
                tables.append((table_text, match.start(), match.end()))
    
    return tables


def semantic_chunks(md_text: str, max_tokens: int = MAX_TOKENS, min_tokens: int = MIN_CHUNK_SIZE) -> List[str]:
    """
    Create semantic chunks using embedding-based boundary detection.
    
    ENHANCED: Tables are preserved as single chunks to maintain data integrity.
    
    1. Extract tables as atomic units
    2. Split remaining text into sentences
    3. Embed each sentence
    4. Find where similarity between adjacent sentences drops
    5. Create chunks at boundaries, respecting max_tokens
    """
    # Extract tables first - they become their own chunks
    tables = extract_markdown_tables(md_text)
    table_chunks = []
    
    if tables:
        print(f"  [semantic] Found {len(tables)} tables to preserve as single chunks")

def summarize_table_context(table_html: str, context_before: str, context_after: str) -> str:
    """
    Use LLM to generate a concise summary of the table based on its context.
    """
    try:
        # Use a cheaper/faster model for summarization if available, otherwise standard
        model_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o") 
        
        prompt = (
            "You are a helpful assistant. Summarize the following table in 1-2 sentences, "
            "explaining what it represents based on the surrounding context. "
            "Focus on the 'who', 'what', and specific conditions/values if critical.\n\n"
            f"--- CONTEXT BEFORE ---\n{context_before[-2000:]}\n\n"
            f"--- TABLE ---\n{table_html[:2000]}\n\n"
            f"--- CONTEXT AFTER ---\n{context_after[:2000]}\n\n"
            "Summary:"
        )
        
        completion = aoai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [summary] Failed to summarize table: {e}")
        return ""


def semantic_chunks(md_text: str, max_tokens: int = MAX_TOKENS, min_tokens: int = MIN_CHUNK_SIZE) -> List[str]:
    """
    Create semantic chunks using embedding-based boundary detection.
    
    ENHANCED: Tables are preserved as single chunks to maintain data integrity.
    LLM summarization allows context injection for better retrieval.
    
    1. Extract tables as atomic units
    2. Split remaining text into sentences
    3. Embed each sentence
    4. Find where similarity between adjacent sentences drops
    5. Create chunks at boundaries, respecting max_tokens
    """
    # Extract tables first - they become their own chunks
    tables = extract_markdown_tables(md_text)
    table_chunks = []
    
    if tables:
        print(f"  [semantic] Found {len(tables)} tables to preserve as single chunks")
        for table_text, start, end in tables:
            # Extract 2500 chars context window
            context_before = md_text[max(0, start-2500):start]
            context_after = md_text[end:min(len(md_text), end+2500)]
            
            # Generate summary via LLM
            print("  [semantic] Generating table summary with LLM...")
            summary = summarize_table_context(table_text, context_before, context_after)
            
            # Combine summary + table
            full_table_text = f"**Context Summary:** {summary}\n\n{table_text}" if summary else table_text
            
            # Check token size
            table_tokens = _count_tokens(full_table_text)
            
            if table_tokens > max_tokens:
                # Table is too large - split by rows but keep header AND summary
                lines = table_text.split('\n')
                header_lines = lines[:2] if len(lines) > 2 else lines[:1]  # Header + separator
                header = '\n'.join(header_lines)
                header_tokens = _count_tokens(header)
                
                # Context is added to every chunk for this table
                context_prefix = f"**Context Summary:** {summary}\n\n" if summary else ""
                context_tokens = _count_tokens(context_prefix)
                
                current_table = context_prefix + header + '\n'
                current_tokens = context_tokens + header_tokens
                
                for line in lines[2:] if len(lines) > 2 else lines[1:]:
                    line_tokens = _count_tokens(line)
                    if current_tokens + line_tokens > max_tokens and current_table != context_prefix + header + '\n':
                        table_chunks.append(current_table.strip())
                        current_table = context_prefix + header + '\n' + line + '\n'
                        current_tokens = context_tokens + header_tokens + line_tokens
                    else:
                        current_table += line + '\n'
                        current_tokens += line_tokens
                
                if current_table.strip() and current_table.strip() != (context_prefix + header).strip():
                    table_chunks.append(current_table.strip())
            else:
                table_chunks.append(full_table_text)
    
    # Remove tables from text for regular processing
    remaining_text = md_text
    for table_text, _, _ in sorted(tables, key=lambda x: x[1], reverse=True):
        remaining_text = remaining_text.replace(table_text, ' [TABLE EXTRACTED] ')
    
    # Process remaining text with semantic chunking
    remaining_text = remaining_text.strip()
    
    if not remaining_text or remaining_text == '[TABLE EXTRACTED]':
        # Only tables, no other content
        print(f"  [semantic] Created {len(table_chunks)} table chunks (no other content)")
        return table_chunks
    
    # Split into sentences
    sentences = split_into_sentences(remaining_text)
    if not sentences:
        # Fallback to simple splitting if no sentences detected
        text_chunks = [remaining_text.strip()] if remaining_text.strip() else []
        all_chunks = table_chunks + text_chunks
        all_chunks = [c for c in all_chunks if c and c != '[TABLE EXTRACTED]']
        return all_chunks
    
    print(f"  [semantic] Split into {len(sentences)} sentences")
    
    # If only a few sentences, just return as single chunk if small enough
    sentence_texts = [s[0] for s in sentences]
    total_tokens = sum(_count_tokens(s) for s in sentence_texts)
    
    if total_tokens <= max_tokens:
        text_chunk = remaining_text.strip().replace('[TABLE EXTRACTED]', '').strip()
        all_chunks = table_chunks + ([text_chunk] if text_chunk else [])
        return all_chunks
    
    # Get embeddings for all sentences
    try:
        embeddings = embed_dense_azure(sentence_texts)
    except Exception as e:
        print(f"  [semantic] Embedding failed: {e}, falling back to simple chunking")
        # Fallback: chunk by token count without semantic boundaries
        text_chunks = _simple_token_chunks(remaining_text, max_tokens)
        all_chunks = table_chunks + text_chunks
        return [c for c in all_chunks if c and c != '[TABLE EXTRACTED]']
    
    # Find semantic boundaries
    boundaries = find_semantic_boundaries(embeddings)
    print(f"  [semantic] Found {len(boundaries)} semantic boundaries")
    
    # Create chunks respecting boundaries and max_tokens
    text_chunks: List[str] = []
    current_chunk_sentences: List[str] = []
    current_tokens = 0
    
    for i, (sentence, _, _) in enumerate(sentences):
        # Skip placeholder sentences
        if '[TABLE EXTRACTED]' in sentence:
            continue
            
        sentence_tokens = _count_tokens(sentence)
        
        # Check if we should break (at semantic boundary OR exceeding max_tokens)
        should_break = False
        if i in boundaries and current_tokens >= min_tokens:
            should_break = True
        elif current_tokens + sentence_tokens > max_tokens and current_chunk_sentences:
            should_break = True
        
        if should_break:
            # Save current chunk
            chunk_text = " ".join(current_chunk_sentences).strip()
            if chunk_text and _count_tokens(chunk_text) >= min_tokens:
                text_chunks.append(chunk_text)
            current_chunk_sentences = []
            current_tokens = 0
        
        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_tokens += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences).strip()
        if chunk_text:
            # If last chunk is too small, merge with previous
            if text_chunks and _count_tokens(chunk_text) < min_tokens:
                text_chunks[-1] = text_chunks[-1] + " " + chunk_text
            else:
                text_chunks.append(chunk_text)
    
    # Combine table chunks and text chunks
    all_chunks = table_chunks + text_chunks
    all_chunks = [c for c in all_chunks if c and c.strip() and '[TABLE EXTRACTED]' not in c]
    
    print(f"  [semantic] Created {len(all_chunks)} chunks ({len(table_chunks)} tables + {len(text_chunks)} text)")
    return all_chunks


def _simple_token_chunks(text: str, max_tokens: int) -> List[str]:
    """Fallback: simple token-based chunking without semantic analysis."""
    words = text.split()
    chunks = []
    current = []
    current_tokens = 0
    
    for word in words:
        word_tokens = _count_tokens(word + " ")
        if current_tokens + word_tokens > max_tokens and current:
            chunks.append(" ".join(current).strip())
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens
    
    if current:
        chunks.append(" ".join(current).strip())
    
    return chunks

# =========================
# Ingest + Search runners (no args)
# =========================
def ingest_folder(pattern: str):
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    dim = infer_embedding_dim()
    ensure_collection(qdrant, COLLECTION_NAME, dim)

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] no markdown files matched: {pattern}")
        return

    total = 0
    for path in files:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        chunks = semantic_chunks(text, MAX_TOKENS, MIN_CHUNK_SIZE)
        if not chunks:
            print(f"[skip] {p.name}: no chunks")
            continue

        dense_vecs  = embed_dense_azure(chunks)     # Azure OAI dense
        sparse_vecs = build_sparse_vectors(chunks)  # BM25 sparse (normalized)
        meta = {"source_file": p.name, "doc_id": p.stem}

        upsert_hybrid_points(qdrant, COLLECTION_NAME, chunks, meta, dense_vecs, sparse_vecs)
        total += len(chunks)
        print(f"[doc] {p.name}: {len(chunks)} chunks")

    print(f"[DONE] Upserted {total} chunks across {len(files)} files → {COLLECTION_NAME}")

def run_hybrid_search(query: str, top_k: int = TOP_K):
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    hits = hybrid_search(qdrant, COLLECTION_NAME, query, top_k=top_k)
    print("=== Hybrid Results ===")
    for i, h in enumerate(hits, 1):
        print(f"{i:>2}. score={h['score']:.4f} | file={h['source_file']} | idx={h['chunk_index']}")
        print(h["preview"])
        print("-" * 80)
    return hits

# =========================
# Main (no args)
# =========================
if __name__ == "__main__":
    if RUN_QUERY is None:
        ingest_folder(INPUT_MD_GLOB)
    else:
        run_hybrid_search(RUN_QUERY, TOP_K)
