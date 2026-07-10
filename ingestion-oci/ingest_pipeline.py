"""
OCI Ingestion Pipeline — main orchestration.

Flow:
  1. PDF → Docling → Markdown + extracted figures
  2. Figures → OCI Gemini 2.5 Pro → text descriptions
  3. Markdown → chunk extraction (7 typed chunks)
  4. Page text → OCI Gemini 2.5 Flash → LLM page grouping
  5. All chunks → OCI Cohere Embed v4.0 → Oracle 26ai AI Vector Search

Replaces: process_single_pdf_multimodal + ingest_single_md_llm
Uses: chunk_types.py, chunk_extractors.py (unchanged from original pipeline)
"""

import json
import logging
import re
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("ingest_pipeline")

# Add pipeline/ to path for chunk_types and chunk_extractors
PIPELINE_DIR = Path(os.getenv("PIPELINE_DIR", str(Path(__file__).parent / "pipeline"))).resolve()
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from chunk_types import ChunkBuilder, ChunkType, count_tokens_rough
from chunk_extractors import (
    parse_doc_filename,
    extract_controls,
    extract_notes,
    extract_figure_blocks,
    extract_markdown_tables,
    clean_ocr_text,
    has_meaningful_content,
    classify_figure_type,
    extract_roles,
    extract_decision_points,
    split_by_pages,
    count_table_rows,
    get_table_header,
)

# Document-Understanding backend: DU_BACKEND=claude routes PDF→Markdown through
# the native Anthropic API (claude_converter — no OCI tenant throttling);
# default remains the Gemini/OCI converter. Both expose the same contract.
if os.getenv("DU_BACKEND", "gemini").lower() == "claude":
    from claude_converter import convert_pdf_to_markdown
else:
    from gemini_converter import convert_pdf_to_markdown
from oci_pipeline import (
    embed_dense_oci,
    get_page_groupings_oci,
)

# Page-grouping backend: GROUPING_BACKEND=claude routes the topic-grouping LLM
# call through native Anthropic (no OCI 429s — this was the throttle-degraded
# step in bulk runs). Aliased onto the same name so call sites stay untouched.
if os.getenv("GROUPING_BACKEND", "gemini").lower() == "claude":
    from claude_converter import get_page_groupings_claude as get_page_groupings_oci

# Split page_context chunks on markdown section headings (## / ###). Requires
# heading-structured markdown (the Claude DU converter emits it); legacy pages
# without headings keep the merged-group behaviour automatically.
SECTION_CHUNKING = os.getenv("SECTION_CHUNKING", "0") == "1"

# Prepend a one-line LLM-generated retrieval-context header to each
# page_context chunk before embedding ("[Context: Employee Attendance policy —
# defines the 'Paid Off' leave type...]"). Bridges vocabulary gaps between how
# users search and how policies are written. Best-effort: header failures never
# block ingestion.
CONTEXT_HEADERS = os.getenv("CONTEXT_HEADERS", "0") == "1"
if CONTEXT_HEADERS:
    from claude_converter import get_section_headers_claude
# Sections smaller than this merge into the previous section — avoids confetti
# chunks from bare headings. Deliberately low: a one-sentence definition like
# "Paid Off" (~90 rough tokens) must survive as its own chunk.
SECTION_MIN_TOKENS = int(os.getenv("SECTION_MIN_TOKENS", "40"))


def _split_sections(text: str):
    """Split markdown into (heading, text) sections on ##/### headings.
    Content before the first heading becomes a leading section with the page
    header (or empty string) as its heading. Undersized sections coalesce into
    their predecessor so we never emit heading-only fragments."""
    lines = text.split("\n")
    raw = []           # list of [heading, [lines]]
    cur_head, cur = "", []
    for ln in lines:
        if re.match(r"^#{2,3}\s+\S", ln):
            if cur or cur_head:
                raw.append([cur_head, cur])
            cur_head, cur = ln.strip(), [ln]
        else:
            cur.append(ln)
    if cur or cur_head:
        raw.append([cur_head, cur])

    # materialize + coalesce small sections into the previous one
    sections = []
    for head, body in raw:
        body_text = "\n".join(body).strip()
        if not body_text:
            continue
        if sections and count_tokens_rough(body_text) < SECTION_MIN_TOKENS:
            sections[-1][1] = sections[-1][1] + "\n\n" + body_text
            continue
        sections.append([head, body_text])
    return [(h, t) for h, t in sections]
import oracle_vectordb
import qdrant_utils
from qdrant_client import QdrantClient

# Live store is Qdrant (we moved off Oracle 26ai). OCI ingestion writes the
# OCI-Cohere-embedded chunks into the same collection the OCI agent reads.
OCI_QDRANT_COLLECTION = os.getenv("OCI_QDRANT_COLLECTION", "docs_oci_ingested_azadea")
_qdrant_client = QdrantClient(url=qdrant_utils.QDRANT_URL, api_key=qdrant_utils.QDRANT_API_KEY,
                              check_compatibility=False)


def process_pdf_oci(
    pdf_path: Path,
    out_dir: Path,
    images_dir: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Convert PDF to markdown using Gemini 2.5 Flash Document Understanding.
    Single API call per page batch — text, tables, and image descriptions all in one pass.

    Returns (md_path, None) — no separate figures JSON needed.
    """
    md_path, _ = convert_pdf_to_markdown(pdf_path, out_dir, images_dir)
    return md_path, None


_SRC_ROOT = Path(os.getenv("PDF_ROOT", "/home/admincsp/multimodal-rag/azadea/data"))
_SRC_EXTS = (".pdf", ".docx", ".txt")


def _original_source_name(md_path: Path) -> str:
    """Resolve the ORIGINAL document filename (with its real extension) for a
    converted markdown file, so chunks store 'X.pdf'/'X.docx' rather than the
    internal 'X.md'. Resolves from disk; falls back to '<stem>.pdf'."""
    stem = md_path.stem
    def _norm(s): return " ".join(str(s).replace("–", "-").replace("—", "-").split()).casefold()
    for e in _SRC_EXTS:                          # exact match first
        hits = list(_SRC_ROOT.rglob(f"{stem}{e}"))
        if hits:
            return hits[0].name
    want = _norm(stem)                            # tolerant (dash/whitespace) match
    for p in _SRC_ROOT.rglob("*"):
        if p.suffix.lower() in _SRC_EXTS and _norm(p.stem) == want:
            return p.name
    return f"{stem}.pdf"


def ingest_md_oci(
    md_path: Path,
    idx: int = 1,
    total: int = 1,
) -> int:
    """
    Steps 3-5: Extract chunks, LLM-group pages, embed, upsert to Oracle 26ai.

    Returns: number of chunks upserted.
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    doc_meta = parse_doc_filename(md_path.name)

    builder = ChunkBuilder(
        source_file=_original_source_name(md_path),
        doc_id=md_path.stem,
        domain=doc_meta.domain,
        function=doc_meta.function,
        variant=doc_meta.variant,
    )

    all_chunks: List[Tuple[str, dict]] = []  # (text_for_embedding, payload)

    # --- Load figure descriptions ---
    figures_json_path = md_path.parent / f"{md_path.stem}_figures.json"
    figure_json = None
    if figures_json_path.exists():
        try:
            figure_json = json.loads(figures_json_path.read_text())
        except Exception as e:
            logger.warning(f"Failed to load figures JSON: {e}")

    # --- CHUNK TYPE 1: image_description ---
    if figure_json:
        for fig in figure_json:
            desc = fig.get("description", "")
            if not desc or len(desc) < 50 or desc.startswith("[Image description"):
                continue
            fig_type = classify_figure_type(desc)
            if fig_type == "logo":
                continue
            roles = extract_roles(desc)
            decision_pts = extract_decision_points(desc)
            has_steps = bool(re.search(r"step\s*\d|steps?\s*:", desc, re.I))
            chunk = builder.build_image_description(
                text=desc,
                page=fig.get("page", 1),
                figure_id=fig.get("id", ""),
                figure_type=fig_type,
                image_path=fig.get("image_path", ""),
                caption=fig.get("caption", ""),
                roles=roles,
                decision_points=decision_pts,
                has_steps=has_steps,
            )
            all_chunks.append((desc, chunk.to_payload()))

    # --- CHUNK TYPE 2: ocr_detail ---
    figure_blocks = extract_figure_blocks(text)
    for i, (fb_text, start, end) in enumerate(figure_blocks):
        ocr_text = clean_ocr_text(fb_text)
        if not has_meaningful_content(ocr_text, min_tokens=50):
            continue
        chunk = builder.build_ocr_detail(text=ocr_text, page=1, figure_idx=i)
        all_chunks.append((ocr_text, chunk.to_payload()))

    # --- CHUNK TYPE 3: control ---
    controls = extract_controls(text)
    for ctrl in controls:
        if len(ctrl.text.strip()) < 20:
            continue
        chunk = builder.build_control(
            text=ctrl.text.strip(),
            page=ctrl.page or 1,
            control_number=ctrl.number,
        )
        all_chunks.append((ctrl.text.strip(), chunk.to_payload()))

    # --- CHUNK TYPE 4: definition ---
    notes = extract_notes(text)
    for note in notes:
        if len(note.text.strip()) < 30:
            continue
        chunk = builder.build_definition(
            text=note.text.strip(),
            page=1,
            note_id=note.note_id,
            definition_terms=note.definition_terms,
        )
        all_chunks.append((note.text.strip(), chunk.to_payload()))

    # --- Strip figures from text for remaining chunk types ---
    text_no_figures = text
    for _, start, end in reversed(figure_blocks):
        text_no_figures = text_no_figures[:start] + " " + text_no_figures[end:]

    # --- CHUNK TYPE 5: table_summary ---
    tables = extract_markdown_tables(text_no_figures)
    for tbl_text, start, end in tables:
        row_count = count_table_rows(tbl_text)
        col_count = 0
        header = get_table_header(tbl_text)
        if header:
            col_count = header.count("|") - 1

        summary = f"Table with {row_count} rows and {col_count} columns. Header: {header}"
        chunk = builder.build_table_summary(
            text=summary,
            page=1,
            table_idx=0,
            row_count=row_count,
            column_count=col_count,
            header=header or "",
            full_table=tbl_text,
        )
        # Embed the summary TOGETHER WITH the full table content so the actual
        # row values (brand names, countries, periods, amounts) are searchable.
        # Embedding only the structural summary ("Table with N rows… Header: …")
        # left every value invisible to dense+sparse search — e.g. a "Zara
        # refund" query could not match a row "Zara | All Countries | 30 Days".
        # full_table stays in the payload (retrieval already surfaces it to the
        # LLM); only the embedded text changes. Guard against a rare oversized
        # table (Cohere Embed v4 allows ~128K tokens, so this is very generous).
        embed_text = f"{summary}\n\n{tbl_text}"
        if count_tokens_rough(embed_text) > 6000:
            embed_text = summary  # fall back for very large tables
        all_chunks.append((embed_text, chunk.to_payload()))

    # Remove tables from text for page_context
    text_clean = text_no_figures
    for tbl_text, start, end in reversed(tables):
        text_clean = text_clean[:start] + " [TABLE_REMOVED] " + text_clean[end:]

    # --- CHUNK TYPE 6: page_context (LLM-grouped, optionally section-split) ---
    pages = split_by_pages(text_clean)
    if not pages:
        pages = {1: text_clean}

    groups, used_fallback = get_page_groupings_oci(text_clean, pages, md_path.stem)
    fallback_tag = " [FALLBACK page-by-page]" if used_fallback else ""
    logger.info(f"[{idx}/{total}] {md_path.stem}: {len(groups)} page groups{fallback_tag}")

    for group in groups:
        merged_parts = []
        for pnum in group:
            if pnum in pages:
                page_text = pages[pnum].strip()
                page_text = page_text.replace("[TABLE_REMOVED]", "").strip()
                if page_text:
                    merged_parts.append(page_text)

        merged_text = "\n\n".join(merged_parts)
        if count_tokens_rough(merged_text) < 100:
            continue

        has_controls = bool(re.search(r"control\s+\d", merged_text, re.I))
        has_notes = "# Notes:" in merged_text or "# Notes" in merged_text
        has_tables = "[TABLE_REMOVED]" in text_no_figures  # original had tables

        # Section-level splitting (SECTION_CHUNKING=1): break the merged group
        # text on markdown section headings (## / ###, produced by the Claude DU
        # converter) so short named definitions — e.g. a leave type defined in
        # one sentence under "### 3.3. Others" — get their own focused chunk
        # instead of drowning in a whole-page blob. Falls back to the single
        # merged chunk when there are no headings (e.g. legacy Gemini markdown).
        sections = _split_sections(merged_text) if SECTION_CHUNKING else []
        if len(sections) <= 1:
            # single merged chunk — treated as one section below
            sections = [("", merged_text)]

        # Contextual retrieval headers: one Claude call covers every section of
        # this group. Empty strings on any failure → chunks ship unenriched.
        headers = ["" for _ in sections]
        if CONTEXT_HEADERS:
            headers = get_section_headers_claude(md_path.stem, sections)

        for s_idx, (s_heading, s_text) in enumerate(sections):
            header = headers[s_idx] if s_idx < len(headers) else ""
            final_text = f"[Context: {header}]\n\n{s_text}" if header else s_text
            chunk = builder.build_page_context(
                text=final_text,
                page=group[0],
                has_controls=bool(re.search(r"control\s+\d", s_text, re.I)),
                has_notes="# Notes" in s_text,
                has_tables=has_tables,
            )
            payload = chunk.to_payload()
            payload["page_group"] = group
            payload["page_start"] = min(group)
            payload["page_end"] = max(group)
            if s_heading:
                payload["section_heading"] = s_heading
            if len(sections) > 1:
                payload["section_index"] = s_idx
            if header:
                payload["context_header"] = header
            all_chunks.append((final_text, payload))

    # --- CHUNK TYPE 7: doc_summary ---
    figure_types = set()
    all_roles = set()
    for _, payload in all_chunks:
        if payload.get("chunk_type") == "image_description":
            ft = payload.get("figure_type", "")
            if ft:
                figure_types.add(ft)
            for r in payload.get("roles", []):
                all_roles.add(r)

    summary_parts = [
        f"Document: {md_path.stem}",
        f"Domain: {doc_meta.domain}" if doc_meta.domain else "",
        f"Function: {doc_meta.function}" if doc_meta.function else "",
        f"Pages: {len(pages)}",
        f"Chunks: {len(all_chunks)}",
    ]
    if figure_types:
        summary_parts.append(f"Figure types: {', '.join(sorted(figure_types))}")
    if all_roles:
        summary_parts.append(f"Roles mentioned: {', '.join(sorted(all_roles))}")

    summary_text = " | ".join(p for p in summary_parts if p)
    chunk = builder.build_doc_summary(text=summary_text, total_pages=len(pages))
    all_chunks.append((summary_text, chunk.to_payload()))

    if not all_chunks:
        logger.warning(f"[{idx}/{total}] {md_path.stem}: no chunks produced")
        return 0

    # --- Embed and upsert ---
    chunk_texts = [t for t, _ in all_chunks]
    chunk_payloads = [p for _, p in all_chunks]

    logger.info(f"[{idx}/{total}] Embedding {len(chunk_texts)} chunks with OCI Cohere Embed v4.0...")
    dense_vecs = embed_dense_oci(chunk_texts)
    sparse_vecs = qdrant_utils.build_sparse_vectors(chunk_texts)

    # Upsert to the live Qdrant collection (dense + sparse, RRF-hybrid ready).
    # Replace any prior chunks for this doc first so update/re-ingest is clean.
    qdrant_utils.delete_by_doc(_qdrant_client, OCI_QDRANT_COLLECTION, md_path.stem)
    inserted = qdrant_utils.upsert_typed_chunks(
        _qdrant_client, OCI_QDRANT_COLLECTION, chunk_payloads, dense_vecs, sparse_vecs
    )

    logger.info(f"[{idx}/{total}] {md_path.stem}: {inserted}/{len(all_chunks)} chunks upserted to Qdrant '{OCI_QDRANT_COLLECTION}'")
    return inserted
