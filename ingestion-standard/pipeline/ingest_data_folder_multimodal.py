#!/usr/bin/env python3
"""
Ingest all PDFs from the data/ folder into Qdrant with MULTIMODAL support.
1. Convert PDFs to markdown using Azure Document Intelligence
2. Extract figures and describe with GPT-4 Vision
3. Ingest markdown files using semantic chunking + hybrid vectors

This is an enhanced version of ingest_data_folder.py with multimodal support.
"""

import logging
import os
import re
import html
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

logger = logging.getLogger("ingest_multimodal")

load_dotenv()

# --- Azure Document Intelligence imports ---
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
    DocumentAnalysisFeature,
    AnalyzeOutputOption,
)
from openai import AzureOpenAI

# --- Import ingestion functions from existing module ---
import azure_doc_intelligence_qdrant as qdrant_ingest
from qdrant_client import QdrantClient

# --- Import multimodal extractor ---
from multimodal_extractor import (
    extract_figures_from_result,
    describe_image_with_gpt4v,
    format_figures_as_markdown,
    get_aoai_client,
)

# ============== CONFIG ==============
DATA_DIR = Path("./data/data")      # Root folder with PDFs
MD_OUT_DIR = Path("./md_out_data_multimodal")  # New output dir for multimodal markdown
USE_HIGHRES = True
LOCALE = None
PAGE_SEPARATOR = "\n\n---\n\n"

# Multimodal settings
ENABLE_MULTIMODAL = True  # Set to False to disable multimodal extraction
MAX_FIGURES_PER_DOC = 10  # Limit figures per document

# Qdrant collection - SEPARATE NAMESPACE for multimodal content
COLLECTION_NAME_MULTIMODAL = "docs_hybrid_azure_azadea_multimodal_updated"

# Parallel processing settings
MAX_WORKERS_PDF = 4   # Reduced for multimodal (more API calls per doc)
MAX_WORKERS_INGEST = 4

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
        raw_cells = [_clean_cell(c) for c in cells]
        # Heuristic: Merge cells that appear split (e.g., "& Bear" should merge with previous)
        merged_cells = []
        if raw_cells:
            merged_cells.append(raw_cells[0])
            for c in raw_cells[1:]:
                if c.startswith("&") and merged_cells:
                    merged_cells[-1] += " " + c
                else:
                    merged_cells.append(c)
        rows.append(merged_cells)
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


def get_operation_id_from_poller(poller) -> Optional[str]:
    """Extract operation ID from poller for figure retrieval."""
    operation_id = None
    
    # Try details dict
    if hasattr(poller, 'details') and poller.details:
        operation_id = poller.details.get('operation_id')
    
    # Try _operation_location URL parsing
    if not operation_id:
        op_location = getattr(poller, '_operation_location', '') or ''
        if '/analyzeResults/' in op_location:
            operation_id = op_location.split('/analyzeResults/')[-1].split('?')[0]
    
    return operation_id


# ============== PDF to Markdown with Multimodal ==============
def process_single_pdf_multimodal(args: Tuple[Path, Path, DocumentIntelligenceClient, AzureOpenAI, int, int]) -> Optional[Path]:
    """Process a single PDF file with multimodal extraction. Returns output path on success."""
    pdf_path, out_dir, doc_client, aoai_client, idx, total = args
    
    try:
        # Prepare analysis request with figure extraction
        features = [DocumentAnalysisFeature.OCR_HIGH_RESOLUTION] if USE_HIGHRES else []
        pdf_bytes = pdf_path.read_bytes()
        body = AnalyzeDocumentRequest(bytes_source=pdf_bytes)
        
        # Include figure extraction if multimodal is enabled
        output_options = [AnalyzeOutputOption.FIGURES] if ENABLE_MULTIMODAL else None
        
        poller = doc_client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=body,
            locale=LOCALE,
            output_content_format=DocumentContentFormat.MARKDOWN,
            features=features,
            output=output_options,
        )
        result = poller.result()
        full = (result.content or "")
        
        # Extract text per page
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
        
        # Build markdown document
        doc_title = pdf_path.stem
        per_doc_blocks = []
        for pnum, page_md in enumerate(per_page_texts, start=1):
            header = f"# {doc_title} — Page {pnum}\n\n> Source file: `{pdf_path.name}` • Page {pnum}\n\n"
            per_doc_blocks.append(header + (page_md or "_(No text recognized on this page)_\n"))
        
        per_doc_markdown = PAGE_SEPARATOR.join(per_doc_blocks).rstrip() + "\n"
        
        # Extract and describe figures if multimodal is enabled
        if ENABLE_MULTIMODAL:
            operation_id = get_operation_id_from_poller(poller)
            
            if operation_id:
                try:
                    figures = extract_figures_from_result(doc_client, result, operation_id, doc_title)
                    
                    if figures:
                        # Limit figures
                        figures = figures[:MAX_FIGURES_PER_DOC]
                        
                        # Describe each figure with GPT-4 Vision
                        figure_metadata = []
                        for fig in figures:
                            if 'image_bytes' in fig:
                                description = describe_image_with_gpt4v(
                                    aoai_client,
                                    fig['image_bytes'],
                                    context=fig.get('caption', ''),
                                    doc_name=doc_title
                                )
                                fig['description'] = description
                                del fig['image_bytes']  # Free memory
                                
                                # Save metadata for Qdrant (includes image_b64)
                                figure_metadata.append({
                                    "id": fig.get('id'),
                                    "caption": fig.get('caption', ''),
                                    "page": fig.get('page', 1),
                                    "description": description,
                                    "image_path": fig.get('image_path', ''),
                                    "image_b64": fig.get('image_b64', ''),
                                })
                        
                        # Save figure metadata to JSON for Qdrant ingestion
                        if figure_metadata:
                            import json
                            figures_json_path = out_dir / f"{doc_title}_figures.json"
                            with open(figures_json_path, 'w', encoding='utf-8') as f:
                                json.dump(figure_metadata, f, indent=2, ensure_ascii=False)
                        
                        # Append figure descriptions to markdown
                        figure_md = format_figures_as_markdown(figures)
                        per_doc_markdown += figure_md
                        logger.info(f"[{idx}/{total}] ✓ {pdf_path.name} (+{len(figures)} figures)")
                    else:
                        logger.info(f"[{idx}/{total}] ✓ {pdf_path.name} (no figures)")
                except Exception as e:
                    logger.error(
                        f"[{idx}/{total}] ⚠ {pdf_path.name}: figure extraction failed: {e}. "
                        "Document will be ingested WITHOUT figure descriptions.",
                        exc_info=True,
                    )
            else:
                logger.warning(f"[{idx}/{total}] {pdf_path.name}: no operation_id — figures skipped")
        else:
            logger.info(f"[{idx}/{total}] ✓ {pdf_path.name}")
        
        # Write output
        out_path = out_dir / f"{doc_title}.md"
        out_path.write_text(per_doc_markdown, encoding="utf-8")
        return out_path
        
    except HttpResponseError as e:
        logger.error(f"[{idx}/{total}] ✗ {pdf_path.name}: Azure DI error: {e}")
        raise RuntimeError(f"Azure Document Intelligence failed for '{pdf_path.name}': {e}") from e
    except Exception as e:
        logger.error(f"[{idx}/{total}] ✗ {pdf_path.name}: {e}", exc_info=True)
        raise RuntimeError(f"PDF→MD conversion failed for '{pdf_path.name}': {e}") from e


def convert_pdfs_multimodal(data_dir: Path, out_dir: Path) -> List[Path]:
    """Convert all PDFs with multimodal extraction."""
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs
    all_pdfs = list(data_dir.rglob("*.pdf"))
    existing_mds = {p.stem for p in out_dir.glob("*.md")}
    pdfs = [p for p in all_pdfs if p.stem not in existing_mds]
    
    print(f"Found {len(all_pdfs)} total PDFs, {len(existing_mds)} already processed, {len(pdfs)} remaining")
    
    if not pdfs:
        print("No new PDFs to process!")
        return list(out_dir.glob("*.md"))
    
    # Create clients
    def create_doc_client():
        return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
    
    aoai_client = get_aoai_client()
    
    # Process PDFs
    total = len(pdfs)
    doc_clients = [create_doc_client() for _ in range(MAX_WORKERS_PDF)]
    args_list = [
        (pdf, out_dir, doc_clients[i % MAX_WORKERS_PDF], aoai_client, i+1, total)
        for i, pdf in enumerate(pdfs)
    ]
    
    output_paths = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PDF) as executor:
        futures = {executor.submit(process_single_pdf_multimodal, args): args[0] for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            if result:
                output_paths.append(result)
    
    print(f"[PDF→MD] Converted {len(output_paths)}/{len(pdfs)} PDFs with multimodal extraction")
    return list(out_dir.glob("*.md"))


# ============== Chunk Extractors ==============
from chunk_extractors import (
    parse_doc_filename,
    classify_figure_type,
    extract_roles,
    extract_decision_points,
    extract_controls,
    extract_notes,
    extract_figure_blocks,
    clean_ocr_text,
    has_meaningful_content,
    has_steps,
    generate_doc_summary_text,
    extract_page_number,
    split_by_pages,
    extract_markdown_tables,
    count_table_rows,
    get_table_header,
    create_table_summary_prompt,
    truncate_table,
)
from chunk_types import (
    ChunkBuilder,
    ChunkType,
    DEFAULT_CHUNK_CONFIG,
    count_tokens_rough,
)


# ============== Qdrant Ingestion ==============
def ingest_single_md_typed(args: Tuple[Path, QdrantClient, str, int, int]) -> int:
    """
    Ingest a single markdown file with TYPED chunks.
    
    Creates multiple chunk types:
        - image_description: From JSON figure descriptions
        - ocr_detail: From <figure> tag content (if meaningful)
        - page_context: Text outside figures
        - control: Individual control statements
        - definition: Notes and definitions
        - doc_summary: Document-level summary
    
    Returns: Total number of chunks created
    """
    import json
    md_path, qdrant_client, collection_name, idx, total = args
    
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        
        # Parse document metadata from filename
        doc_meta = parse_doc_filename(md_path.name)
        
        # Create chunk builder with document metadata
        builder = ChunkBuilder(
            source_file=md_path.name,
            doc_id=md_path.stem,
            domain=doc_meta.domain,
            function=doc_meta.function,
            variant=doc_meta.variant,
        )
        
        all_chunks = []  # List of (text, payload) tuples
        
        # Load figure JSON if exists
        figures_json_path = md_path.parent / f"{md_path.stem}_figures.json"
        figure_json = None
        if figures_json_path.exists():
            try:
                with open(figures_json_path, 'r', encoding='utf-8') as f:
                    figure_json = json.load(f)
            except Exception as e:
                logger.warning(f"[{idx}/{total}] Failed to load figures JSON: {e}")
        
        # =====================================================
        # CHUNK TYPE 1: Image Description Chunks (from JSON)
        # =====================================================
        if figure_json:
            for fig_data in figure_json:
                desc = fig_data.get('description', '')
                if not desc or len(desc) < 50:
                    continue
                
                fig_type = classify_figure_type(desc)
                
                # Skip logos (noise)
                if fig_type == 'logo':
                    continue
                
                chunk = builder.build_image_description(
                    text=desc,
                    page=fig_data.get('page', 1),
                    figure_id=str(fig_data.get('id', '')),
                    figure_type=fig_type,
                    image_path=fig_data.get('image_path', ''),
                    caption=fig_data.get('caption', ''),
                    roles=extract_roles(desc),
                    decision_points=extract_decision_points(desc),
                    has_steps=has_steps(desc),
                )
                all_chunks.append((desc, chunk.to_payload()))
        
        # =====================================================
        # CHUNK TYPE 2: OCR Detail Chunks (from <figure> tags)
        # =====================================================
        figure_blocks = extract_figure_blocks(text)
        for fig_idx, (fig_content, start, end) in enumerate(figure_blocks):
            ocr_text = clean_ocr_text(fig_content)
            
            # Skip if not meaningful (too short)
            if not has_meaningful_content(ocr_text, min_tokens=50):
                continue
            
            # Find page number from context
            page = extract_page_number(text[max(0, start-200):start]) or 1
            
            chunk = builder.build_ocr_detail(
                text=ocr_text,
                page=page,
                figure_idx=fig_idx,
                roles=extract_roles(ocr_text),
                decision_points=extract_decision_points(ocr_text),
                has_steps=has_steps(ocr_text),
            )
            all_chunks.append((ocr_text, chunk.to_payload()))
        
        # =====================================================
        # CHUNK TYPE 3: Control Chunks
        # =====================================================
        controls = extract_controls(text)
        for ctrl in controls:
            if len(ctrl.text) < 20:
                continue
            
            chunk = builder.build_control(
                text=ctrl.text,
                page=ctrl.page or 1,
                control_number=ctrl.number,
            )
            all_chunks.append((ctrl.text, chunk.to_payload()))
        
        # =====================================================
        # CHUNK TYPE 4: Definition/Notes Chunks
        # =====================================================
        notes = extract_notes(text)
        for note in notes:
            if len(note.text) < 30:
                continue
            
            chunk = builder.build_definition(
                text=note.text,
                page=note.page or 1,
                note_id=note.note_id,
                definition_terms=note.definition_terms,
            )
            all_chunks.append((note.text, chunk.to_payload()))
        
        # =====================================================
        # CHUNK TYPE 5: Page Context Chunks (text outside figures)
        # =====================================================
        # Remove figures from text
        text_without_figures = text
        for fig_content, start, end in sorted(figure_blocks, key=lambda x: x[1], reverse=True):
            text_without_figures = text_without_figures[:start] + ' ' + text_without_figures[end:]
        
        # Max tokens for embedding model (Azure text-embedding-3-large = 8192)
        MAX_EMBED_TOKENS = 7500  # Leave buffer for safety
        
        # =====================================================
        # CHUNK TYPE 5a: Table Summary Chunks
        # =====================================================
        # Extract tables and create summary chunks (NOT full table content)
        tables = extract_markdown_tables(text_without_figures)
        table_placeholder = "[TABLE_REMOVED]"
        
        for table_idx, (table_text, table_start, table_end) in enumerate(tables):
            row_count = count_table_rows(table_text)
            header = get_table_header(table_text)
            
            # Count columns from header
            col_count = header.count('|') - 1 if header else 0
            
            # Create summary text for EMBEDDING (header + metadata, NOT full table)
            summary_text = f"Table with {row_count} rows and {col_count} columns. Header: {header}"
            
            # Find which page this table is on
            page = extract_page_number(text_without_figures[max(0, table_start-500):table_start]) or 1
            
            # Store full table in metadata (for RAG to use complete content)
            chunk = builder.build_table_summary(
                text=summary_text,  # This gets embedded
                page=page,
                table_idx=table_idx,
                row_count=row_count,
                column_count=col_count,
                header=header,
                full_table=table_text,  # Complete table stored in metadata
            )
            all_chunks.append((summary_text, chunk.to_payload()))
        
        # Remove tables from text for page_context (replace with placeholder)
        text_for_pages = text_without_figures
        for table_text, start, end in sorted(tables, key=lambda x: x[1], reverse=True):
            text_for_pages = text_for_pages[:start] + f" {table_placeholder} " + text_for_pages[end:]
        
        # =====================================================
        # CHUNK TYPE 5b: Page Context Chunks (without tables)
        # =====================================================
        pages = split_by_pages(text_for_pages)
        for page_num, page_text in pages.items():
            # Clean up the page text
            page_text = re.sub(r'\s+', ' ', page_text).strip()
            # Remove table placeholders
            page_text = page_text.replace(table_placeholder, '').strip()
            page_text = re.sub(r'\s+', ' ', page_text).strip()
            
            # Skip very short pages or pages that are mostly just headers
            if count_tokens_rough(page_text) < 100:
                continue
            
            # Check for controls and notes
            has_ctrl = 'Control' in page_text
            has_note = 'Notes:' in page_text or '(' in page_text
            has_table = False  # Tables are now separate chunks
            
            chunk = builder.build_page_context(
                text=page_text,
                page=page_num,
                has_controls=has_ctrl,
                has_notes=has_note,
                has_tables=has_table,
            )
            all_chunks.append((page_text, chunk.to_payload()))
        
        # =====================================================
        # CHUNK TYPE 6: Document Summary Chunk
        # =====================================================
        # Collect metadata for summary
        figure_types = list(set(
            classify_figure_type(f.get('description', ''))
            for f in (figure_json or [])
            if classify_figure_type(f.get('description', '')) != 'logo'
        ))
        
        all_roles = set()
        for _, payload in all_chunks:
            all_roles.update(payload.get('roles', []))
        
        summary_text = f"Document: {doc_meta.title or md_path.stem}. "
        if doc_meta.domain and doc_meta.function:
            summary_text += f"Department: {doc_meta.domain}, Function: {doc_meta.function}. "
        if figure_types:
            summary_text += f"Contains: {', '.join(figure_types)}. "
        if all_roles:
            summary_text += f"Key roles: {', '.join(list(all_roles)[:5])}. "
        if controls:
            summary_text += f"Includes {len(controls)} control points. "
        
        total_pages = max(pages.keys()) if pages else 1
        summary_text += f"Pages: {total_pages}."
        
        summary_chunk = builder.build_doc_summary(
            text=summary_text,
            total_pages=total_pages,
            figure_types=figure_types,
            key_roles=list(all_roles)[:10],
            control_count=len(controls),
        )
        all_chunks.append((summary_text, summary_chunk.to_payload()))
        
        # =====================================================
        # EMBED AND UPSERT
        # =====================================================
        if not all_chunks:
            logger.warning(f"[{idx}/{total}] skip {md_path.name}: no chunks produced")
            return 0

        # Extract texts for embedding
        chunk_texts = [t for t, _ in all_chunks]
        chunk_payloads = [p for _, p in all_chunks]

        # Generate embeddings
        dense_vecs = qdrant_ingest.embed_dense_azure(chunk_texts)
        sparse_vecs = qdrant_ingest.build_sparse_vectors(chunk_texts)

        # Upsert typed chunks
        qdrant_ingest.upsert_typed_chunks(
            qdrant_client, collection_name,
            chunk_payloads, dense_vecs, sparse_vecs
        )

        # Summary
        type_counts = {}
        for _, p in all_chunks:
            ct = p.get('chunk_type', 'unknown')
            type_counts[ct] = type_counts.get(ct, 0) + 1

        type_str = ', '.join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
        logger.info(f"[{idx}/{total}] ✓ {md_path.name}: {len(all_chunks)} chunks ({type_str})")
        return len(all_chunks)

    except Exception as e:
        logger.error(f"[{idx}/{total}] ✗ {md_path.name}: {e}", exc_info=True)
        raise RuntimeError(f"Typed ingestion failed for '{md_path.name}': {e}") from e


# Legacy function for backward compatibility
def ingest_single_md(args: Tuple[Path, QdrantClient, str, int, int]) -> int:
    """Legacy ingestion - redirects to typed ingestion."""
    return ingest_single_md_typed(args)


def ingest_markdown_parallel(md_files: List[Path]):
    """Ingest all markdown files into the SEPARATE multimodal Qdrant collection."""
    print(f"Ingesting {len(md_files)} markdown files into Qdrant collection: {COLLECTION_NAME_MULTIMODAL}")
    
    qdrant_client = QdrantClient(url=qdrant_ingest.QDRANT_URL, api_key=qdrant_ingest.QDRANT_API_KEY)
    dim = qdrant_ingest.infer_embedding_dim()
    
    # Create/ensure the SEPARATE multimodal collection
    qdrant_ingest.ensure_collection(qdrant_client, COLLECTION_NAME_MULTIMODAL, dim)
    
    total = len(md_files)
    args_list = [(md, qdrant_client, COLLECTION_NAME_MULTIMODAL, i+1, total) for i, md in enumerate(md_files)]
    
    total_chunks = 0
    failed_files = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_INGEST) as executor:
        futures = {executor.submit(ingest_single_md, args): args[0] for args in args_list}
        for future in as_completed(futures):
            md_file = futures[future]
            try:
                total_chunks += future.result()
            except Exception as e:
                print(f"[BATCH] FAILED {md_file.name}: {e}")
                failed_files.append(md_file.name)

    print(f"[QDRANT] Ingested {total_chunks} chunks from {len(md_files)} files → {COLLECTION_NAME_MULTIMODAL}")
    if failed_files:
        print(f"FAILED ({len(failed_files)}): {', '.join(failed_files)}")


def main():
    print("=" * 60)
    print("MULTIMODAL PDF INGESTION")
    print(f"Multimodal extraction: {'ENABLED' if ENABLE_MULTIMODAL else 'DISABLED'}")
    print(f"Qdrant collection: {COLLECTION_NAME_MULTIMODAL}")
    print("=" * 60)
    
    print("\n=== Step 1: Convert PDFs to Markdown with Figures ===")
    md_files = convert_pdfs_multimodal(DATA_DIR, MD_OUT_DIR)
    
    print("\n=== Step 2: Ingest Markdown to Qdrant ===")
    ingest_markdown_parallel(md_files)
    
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
