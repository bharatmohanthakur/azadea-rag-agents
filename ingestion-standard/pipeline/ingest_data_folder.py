#!/usr/bin/env python3
"""
Ingest all PDFs from the data/ folder into Qdrant with PARALLEL PROCESSING.
1. Convert PDFs to markdown using Azure Document Intelligence (parallel)
2. Ingest markdown files using semantic chunking + hybrid vectors (parallel)
"""

import os
import re
import html
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# --- Azure Document Intelligence imports ---
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
    DocumentAnalysisFeature,
)

# --- Import ingestion functions from existing module ---
import azure_doc_intelligence_qdrant as qdrant_ingest
from qdrant_client import QdrantClient

# ============== CONFIG ==============
DATA_DIR = Path("./data/data")      # Root folder with PDFs
MD_OUT_DIR = Path("./md_out_data")  # Markdown output for data folder
USE_HIGHRES = True
LOCALE = None
PAGE_SEPARATOR = "\n\n---\n\n"

# Parallel processing settings
MAX_WORKERS_PDF = 8   # Parallel PDF conversions (Azure API calls)
MAX_WORKERS_INGEST = 4  # Parallel Qdrant ingestion

# ============== Table converter ==============
_TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TH_TD_RE = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")

def _clean_cell(text: str) -> str:
    text = html.unescape(text)
    text = _TAG_RE.sub("", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text).strip()
    text = text.replace("\n", " ").replace("|", "\\|")
    return text

def _table_html_to_markdown(table_html: str) -> str:
    rows_html = _TR_RE.findall(table_html)
    if not rows_html:
        return table_html
    rows = []
    for rhtml in rows_html:
        cells = _TH_TD_RE.findall(rhtml)
        if not cells:
            cells = re.split(r"</?t[hd][^>]*>", rhtml, flags=re.IGNORECASE)
            cells = [c for c in cells if c.strip()]
        rows.append([_clean_cell(c) for c in cells])
    max_cols = max((len(r) for r in rows), default=0)
    if max_cols == 0:
        return table_html
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    header = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []
    if not any(cell.strip() for cell in header):
        header = [f"Col {i+1}" for i in range(max_cols)]
        body = rows
    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for r in body:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)

def convert_xml_tables_to_markdown(markdown_or_html: str) -> str:
    def _repl(m):
        inner = m.group(1)
        try:
            return "\n" + _table_html_to_markdown(inner) + "\n"
        except Exception:
            return m.group(0)
    return _TABLE_RE.sub(_repl, markdown_or_html)

# ============== PDF to Markdown (single file) ==============
def process_single_pdf(args: Tuple[Path, Path, DocumentIntelligenceClient, int, int]) -> Optional[Path]:
    """Process a single PDF file. Returns output path on success, None on failure."""
    pdf_path, out_dir, client, idx, total = args
    try:
        features = [DocumentAnalysisFeature.OCR_HIGH_RESOLUTION] if USE_HIGHRES else []
        pdf_bytes = pdf_path.read_bytes()
        body = AnalyzeDocumentRequest(bytes_source=pdf_bytes)
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=body,
            locale=LOCALE,
            output_content_format=DocumentContentFormat.MARKDOWN,
            features=features,
        )
        result = poller.result()
        full = (result.content or "")
        
        per_page_texts: List[str] = []
        if getattr(result, "pages", None):
            for p in result.pages:
                page_fragments = []
                spans = getattr(p, "spans", None) or []
                for s in spans:
                    start = getattr(s, "offset", 0)
                    length = getattr(s, "length", 0)
                    if length > 0 and 0 <= start < len(full):
                        page_fragments.append(full[start:start+length])
                page_text = "".join(page_fragments).strip()
                page_text = convert_xml_tables_to_markdown(page_text)
                if not page_text.endswith("\n"):
                    page_text += "\n"
                per_page_texts.append(page_text)
        else:
            chunk = convert_xml_tables_to_markdown(full.strip())
            if not chunk.endswith("\n"):
                chunk += "\n"
            per_page_texts = [chunk]
        
        doc_title = pdf_path.stem
        per_doc_blocks = []
        for pnum, page_md in enumerate(per_page_texts, start=1):
            header = f"# {doc_title} — Page {pnum}\n\n> Source file: `{pdf_path.name}` • Page {pnum}\n\n"
            per_doc_blocks.append(header + (page_md or "_(No text recognized on this page)_\n"))
        
        per_doc_markdown = PAGE_SEPARATOR.join(per_doc_blocks).rstrip() + "\n"
        out_path = out_dir / f"{doc_title}.md"
        out_path.write_text(per_doc_markdown, encoding="utf-8")
        print(f"[{idx}/{total}] ✓ {pdf_path.name}")
        return out_path
    except HttpResponseError as e:
        print(f"[{idx}/{total}] ✗ {pdf_path.name}: {e}")
        return None
    except Exception as e:
        print(f"[{idx}/{total}] ✗ {pdf_path.name}: {e}")
        return None

def convert_pdfs_parallel(data_dir: Path, out_dir: Path) -> List[Path]:
    """Convert all PDFs in data_dir (recursively) to markdown files in out_dir using parallel processing."""
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
    api_key  = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY") or os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs recursively, skip already processed
    all_pdfs = list(data_dir.rglob("*.pdf"))
    existing_mds = {p.stem for p in out_dir.glob("*.md")}
    pdfs = [p for p in all_pdfs if p.stem not in existing_mds]
    
    print(f"Found {len(all_pdfs)} total PDFs, {len(existing_mds)} already processed, {len(pdfs)} remaining")
    
    if not pdfs:
        print("No new PDFs to process!")
        return list(out_dir.glob("*.md"))
    
    # Create one client per worker (thread-safe)
    def create_client():
        return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
    
    # Prepare args
    total = len(pdfs)
    clients = [create_client() for _ in range(MAX_WORKERS_PDF)]
    args_list = [(pdf, out_dir, clients[i % MAX_WORKERS_PDF], i+1, total) for i, pdf in enumerate(pdfs)]
    
    output_paths = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PDF) as executor:
        futures = {executor.submit(process_single_pdf, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            if result:
                output_paths.append(result)
    
    print(f"[PDF→MD] Converted {len(output_paths)}/{len(pdfs)} PDFs successfully")
    return list(out_dir.glob("*.md"))

# ============== Qdrant Ingestion (parallel) ==============
def ingest_single_md(args: Tuple[Path, QdrantClient, int, int]) -> int:
    """Ingest a single markdown file. Returns number of chunks."""
    md_path, qdrant_client, idx, total = args
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        chunks = qdrant_ingest.semantic_chunks(text, qdrant_ingest.MAX_TOKENS, qdrant_ingest.MIN_CHUNK_SIZE)
        if not chunks:
            print(f"[{idx}/{total}] skip {md_path.name}: no chunks")
            return 0
        
        dense_vecs = qdrant_ingest.embed_dense_azure(chunks)
        sparse_vecs = qdrant_ingest.build_sparse_vectors(chunks)
        meta = {"source_file": md_path.name, "doc_id": md_path.stem}
        
        qdrant_ingest.upsert_hybrid_points(qdrant_client, qdrant_ingest.COLLECTION_NAME, chunks, meta, dense_vecs, sparse_vecs)
        print(f"[{idx}/{total}] ✓ {md_path.name}: {len(chunks)} chunks")
        return len(chunks)
    except Exception as e:
        print(f"[{idx}/{total}] ✗ {md_path.name}: {e}")
        return 0

def ingest_markdown_parallel(md_files: List[Path]):
    """Ingest all markdown files into Qdrant using parallel processing."""
    print(f"Ingesting {len(md_files)} markdown files into Qdrant...")
    
    qdrant_client = QdrantClient(url=qdrant_ingest.QDRANT_URL, api_key=qdrant_ingest.QDRANT_API_KEY)
    dim = qdrant_ingest.infer_embedding_dim()
    qdrant_ingest.ensure_collection(qdrant_client, qdrant_ingest.COLLECTION_NAME, dim)
    
    total = len(md_files)
    args_list = [(md, qdrant_client, i+1, total) for i, md in enumerate(md_files)]
    
    total_chunks = 0
    # Use smaller parallelism for embedding (Azure rate limits)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_INGEST) as executor:
        futures = {executor.submit(ingest_single_md, args): args[0] for args in args_list}
        for future in as_completed(futures):
            total_chunks += future.result()
    
    print(f"[QDRANT] Ingested {total_chunks} chunks from {len(md_files)} files")

def main():
    print("=== Step 1: Convert PDFs to Markdown (Parallel) ===")
    md_files = convert_pdfs_parallel(DATA_DIR, MD_OUT_DIR)
    
    print("\n=== Step 2: Ingest Markdown to Qdrant (Parallel) ===")
    ingest_markdown_parallel(md_files)
    
    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
