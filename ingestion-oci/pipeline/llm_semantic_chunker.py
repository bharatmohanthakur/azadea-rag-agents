#!/usr/bin/env python3
"""
LLM-Guided Semantic Chunking for Multimodal RAG.

Instead of splitting documents page-by-page, this module sends the full document
to GPT-4o and asks it which pages belong together topically. The result is
semantically coherent chunks that may span multiple pages — improving retrieval
accuracy because each chunk is a self-contained topic.

All other chunk types (image_description, ocr_detail, control, definition,
table_summary, doc_summary) remain unchanged.

Hybrid search: Dense (Azure text-embedding-3-large) + Sparse (BM25/fastembed)
with server-side RRF fusion — same as existing pipeline.

Usage:
    python llm_semantic_chunker.py                  # Re-ingest all docs
    python llm_semantic_chunker.py --dry-run        # Preview groupings only
    python llm_semantic_chunker.py --file "HRD*"    # Process matching files
"""

import logging
import os
import re
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

logger = logging.getLogger("llm_semantic_chunker")

load_dotenv()

from openai import AzureOpenAI

# --- Imports from existing modules ---
import azure_doc_intelligence_qdrant as qdrant_ingest
from qdrant_client import QdrantClient
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
    extract_page_number,
    split_by_pages,
    extract_markdown_tables,
    count_table_rows,
    get_table_header,
)
from chunk_types import ChunkBuilder, ChunkType, count_tokens_rough

# ============== CONFIG ==============
MD_DIR = Path("./md_out_data_multimodal")

# NEW separate collection — does NOT touch the existing one
COLLECTION_NAME = "docs_llm_chunked_azadea"

# Azure OpenAI for page grouping
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
LLM_DEPLOYMENT = "gpt-4o"  # GPT-4o for page grouping

# LLM client
aoai = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01",
)


# =============================================================================
# LLM PAGE GROUPING
# =============================================================================

def get_page_groupings(doc_text: str, pages: Dict[int, str], doc_name: str) -> Tuple[List[List[int]], bool]:
    """
    Send FULL document to GPT-4o and ask which pages belong together topically.

    Returns:
        (groups, used_fallback) — groups is a list of page groups e.g. [[1,2,3], [4,5], [6]].
        used_fallback is True if LLM calls failed and page-by-page fallback was used.
    """
    total_pages = max(pages.keys()) if pages else 1

    # Single page → trivial grouping
    if total_pages <= 1:
        return [[1]], False

    # Build full page content — NO truncation
    page_sections = []
    for pnum in sorted(pages.keys()):
        page_text = pages[pnum].strip()
        page_sections.append(f"=== PAGE {pnum} ===\n{page_text}")

    combined = "\n\n".join(page_sections)

    prompt = f"""You are analyzing an HR policy document to determine which pages discuss the same topic.

Document: {doc_name}
Total pages: {total_pages}

Below is the full content of each page. Group pages that cover the same topic or section together. Pages in the same group will be merged into a single retrieval chunk.

Rules:
- Every page number (1 to {total_pages}) must appear in exactly one group
- Keep groups topically coherent — pages about the same policy section go together
- A group can be a single page if that page covers a distinct topic
- Maximum 5 pages per group (to keep chunks reasonable for retrieval)
- Output ONLY the groupings in this exact format, one per line:
  GROUP: 1,2,3 | Topic name
  GROUP: 4,5 | Topic name
  GROUP: 6 | Topic name

{combined}"""

    for attempt in range(3):
        try:
            response = aoai.chat.completions.create(
                model=LLM_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            raw = response.choices[0].message.content.strip()
            groups = _parse_grouping_response(raw, total_pages)
            return groups, False
        except Exception as e:
            sleep_time = 2 ** attempt
            logger.warning(f"[llm-chunk] retry {attempt+1}/3 for '{doc_name}': {e} (sleep {sleep_time}s)")
            time.sleep(sleep_time)

    # Fallback: one page per group — LLM grouping failed
    logger.warning(f"[llm-chunk] FALLBACK: page-by-page for '{doc_name}' after 3 failed LLM attempts")
    return [[p] for p in sorted(pages.keys())], True


def _parse_grouping_response(raw: str, total_pages: int) -> List[List[int]]:
    """
    Parse LLM response like:
        GROUP: 1,2,3 | Leave Eligibility
        GROUP: 4,5 | Carry Over Policy

    Returns [[1,2,3], [4,5], ...]
    Validates all pages 1..total_pages are covered.
    """
    groups = []
    seen = set()

    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Extract page numbers — look for digits after GROUP:
        match = re.search(r'GROUP\s*:\s*([\d,\s]+)', line, re.IGNORECASE)
        if not match:
            # Try looser pattern: just digits at start of line
            match = re.match(r'^([\d,\s]+)', line)

        if match:
            nums_str = match.group(1)
            page_nums = []
            for n in re.findall(r'\d+', nums_str):
                pn = int(n)
                if 1 <= pn <= total_pages and pn not in seen:
                    page_nums.append(pn)
                    seen.add(pn)
            if page_nums:
                groups.append(sorted(page_nums))

    # Fill in any missing pages as individual groups
    for p in range(1, total_pages + 1):
        if p not in seen:
            groups.append([p])

    return groups


# =============================================================================
# INGESTION WITH LLM-GUIDED CHUNKS
# =============================================================================

def ingest_single_md_llm(args: Tuple[Path, QdrantClient, str, int, int, bool]) -> int:
    """
    Ingest a single markdown file using LLM-guided page grouping.

    Hybrid vectors: Dense (Azure text-embedding-3-large) + Sparse (BM25/fastembed).
    Same as existing pipeline but replaces page_context chunking
    with LLM-determined page groupings.
    """
    md_path, qdrant_client, collection_name, idx, total, dry_run = args

    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        doc_meta = parse_doc_filename(md_path.name)

        builder = ChunkBuilder(
            source_file=md_path.name,
            doc_id=md_path.stem,
            domain=doc_meta.domain,
            function=doc_meta.function,
            variant=doc_meta.variant,
        )

        all_chunks = []  # (text, payload) tuples

        # --- Load figure JSON ---
        figures_json_path = md_path.parent / f"{md_path.stem}_figures.json"
        figure_json = None
        if figures_json_path.exists():
            try:
                with open(figures_json_path, "r", encoding="utf-8") as f:
                    figure_json = json.load(f)
            except Exception as e:
                logger.warning(f"[{idx}/{total}] figures JSON parse error: {e}")

        # =====================================================
        # CHUNK TYPE 1: Image Description (unchanged)
        # =====================================================
        if figure_json:
            for fig_data in figure_json:
                desc = fig_data.get("description", "")
                if not desc or len(desc) < 50:
                    continue
                fig_type = classify_figure_type(desc)
                if fig_type == "logo":
                    continue
                chunk = builder.build_image_description(
                    text=desc,
                    page=fig_data.get("page", 1),
                    figure_id=str(fig_data.get("id", "")),
                    figure_type=fig_type,
                    image_path=fig_data.get("image_path", ""),
                    caption=fig_data.get("caption", ""),
                    roles=extract_roles(desc),
                    decision_points=extract_decision_points(desc),
                    has_steps=has_steps(desc),
                )
                all_chunks.append((desc, chunk.to_payload()))

        # =====================================================
        # CHUNK TYPE 2: OCR Detail (unchanged)
        # =====================================================
        figure_blocks = extract_figure_blocks(text)
        for fig_idx, (fig_content, start, end) in enumerate(figure_blocks):
            ocr_text = clean_ocr_text(fig_content)
            if not has_meaningful_content(ocr_text, min_tokens=50):
                continue
            page = extract_page_number(text[max(0, start - 200) : start]) or 1
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
        # CHUNK TYPE 3: Controls (unchanged)
        # =====================================================
        controls = extract_controls(text)
        for ctrl in controls:
            if len(ctrl.text) < 20:
                continue
            chunk = builder.build_control(
                text=ctrl.text, page=ctrl.page or 1, control_number=ctrl.number
            )
            all_chunks.append((ctrl.text, chunk.to_payload()))

        # =====================================================
        # CHUNK TYPE 4: Definitions/Notes (unchanged)
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
        # Prepare text for page_context (remove figures)
        # =====================================================
        text_without_figures = text
        for fig_content, start, end in sorted(
            figure_blocks, key=lambda x: x[1], reverse=True
        ):
            text_without_figures = (
                text_without_figures[:start] + " " + text_without_figures[end:]
            )

        # =====================================================
        # CHUNK TYPE 5a: Table Summary (unchanged)
        # =====================================================
        tables = extract_markdown_tables(text_without_figures)
        table_placeholder = "[TABLE_REMOVED]"

        for table_idx, (table_text, table_start, table_end) in enumerate(tables):
            row_count = count_table_rows(table_text)
            header = get_table_header(table_text)
            col_count = header.count("|") - 1 if header else 0
            summary_text = (
                f"Table with {row_count} rows and {col_count} columns. Header: {header}"
            )
            page = (
                extract_page_number(
                    text_without_figures[max(0, table_start - 500) : table_start]
                )
                or 1
            )
            chunk = builder.build_table_summary(
                text=summary_text,
                page=page,
                table_idx=table_idx,
                row_count=row_count,
                column_count=col_count,
                header=header,
                full_table=table_text,
            )
            # Embed the summary TOGETHER WITH the full table content so the
            # actual row values (brand names, countries, periods, amounts) are
            # searchable. Embedding only the structural summary
            # ("Table with N rows… Header: …") left every value invisible to
            # both dense and sparse search — e.g. a "Zara refund" query could
            # not match a row reading "Zara | All Countries | 30 Days".
            # full_table stays in the payload (the retrieval tool already
            # surfaces it to the LLM); only the embedded text changes here.
            # Guard against a rare oversized table blowing the embedding
            # model's token budget (text-embedding-3-large = 8192).
            embed_text = f"{summary_text}\n\n{table_text}"
            if qdrant_ingest._count_tokens(embed_text) > 6000:
                embed_text = summary_text  # fall back for very large tables
            all_chunks.append((embed_text, chunk.to_payload()))

        # Remove tables from text
        text_for_pages = text_without_figures
        for table_text, start, end in sorted(tables, key=lambda x: x[1], reverse=True):
            text_for_pages = (
                text_for_pages[:start]
                + f" {table_placeholder} "
                + text_for_pages[end:]
            )

        # =====================================================
        # CHUNK TYPE 5b: LLM-GUIDED Page Context (THE NEW PART)
        # =====================================================
        pages = split_by_pages(text_for_pages)

        used_fallback = False
        if pages:
            # Ask LLM to group pages — sends FULL page content, no truncation
            groupings, used_fallback = get_page_groupings(text_for_pages, pages, md_path.stem)

            topic_labels = []
            for group in groupings:
                label = f"pages {','.join(str(p) for p in group)}"
                topic_labels.append(label)

            fallback_tag = " [FALLBACK page-by-page]" if used_fallback else ""
            logger.info(
                f"[{idx}/{total}] LLM groups: {' | '.join(topic_labels)}{fallback_tag}"
            )

            if dry_run:
                return 0

            for group in groupings:
                # Merge page texts for this group
                merged_parts = []
                for pnum in sorted(group):
                    if pnum in pages:
                        merged_parts.append(pages[pnum])

                merged_text = "\n".join(merged_parts)
                # Clean
                merged_text = re.sub(r"\s+", " ", merged_text).strip()
                merged_text = merged_text.replace(table_placeholder, "").strip()
                merged_text = re.sub(r"\s+", " ", merged_text).strip()

                if count_tokens_rough(merged_text) < 100:
                    continue

                first_page = min(group)
                has_ctrl = "Control" in merged_text
                has_note = "Notes:" in merged_text or "(" in merged_text

                chunk = builder.build_page_context(
                    text=merged_text,
                    page=first_page,
                    has_controls=has_ctrl,
                    has_notes=has_note,
                    has_tables=False,
                )

                # Add extra metadata about the page span
                payload = chunk.to_payload()
                payload["page_group"] = group
                payload["page_start"] = min(group)
                payload["page_end"] = max(group)
                all_chunks.append((merged_text, payload))

        # =====================================================
        # CHUNK TYPE 6: Document Summary (unchanged)
        # =====================================================
        figure_types = list(
            set(
                classify_figure_type(f.get("description", ""))
                for f in (figure_json or [])
                if classify_figure_type(f.get("description", "")) != "logo"
            )
        )

        all_roles = set()
        for _, payload in all_chunks:
            all_roles.update(payload.get("roles", []))

        summary_text = f"Document: {doc_meta.title or md_path.stem}. "
        if doc_meta.domain and doc_meta.function:
            summary_text += (
                f"Department: {doc_meta.domain}, Function: {doc_meta.function}. "
            )
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
        # EMBED AND UPSERT — Hybrid: Dense + Sparse (BM25)
        # =====================================================
        if not all_chunks:
            logger.warning(f"[{idx}/{total}] skip {md_path.name}: no chunks produced")
            return 0

        chunk_texts = [t for t, _ in all_chunks]
        chunk_payloads = [p for _, p in all_chunks]

        # Dense: Azure OpenAI text-embedding-3-large
        dense_vecs = qdrant_ingest.embed_dense_azure(chunk_texts)
        # Sparse: BM25 via fastembed
        sparse_vecs = qdrant_ingest.build_sparse_vectors(chunk_texts)

        # Upsert with BOTH vectors for hybrid RRF search
        qdrant_ingest.upsert_typed_chunks(
            qdrant_client, collection_name, chunk_payloads, dense_vecs, sparse_vecs
        )

        type_counts = {}
        for _, p in all_chunks:
            ct = p.get("chunk_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1

        type_str = ", ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
        logger.info(f"[{idx}/{total}] done {md_path.name}: {len(all_chunks)} chunks ({type_str})")
        return len(all_chunks)

    except Exception as e:
        logger.error(f"[{idx}/{total}] ERROR {md_path.name}: {e}", exc_info=True)
        raise RuntimeError(f"Ingestion failed for '{md_path.name}': {e}") from e


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-guided semantic chunking")
    parser.add_argument("--dry-run", action="store_true", help="Preview groupings only, no ingestion")
    parser.add_argument("--file", type=str, default=None, help="Glob pattern to filter files (e.g. 'HRD*')")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--clear", action="store_true", help="Clear collection before ingesting")
    args = parser.parse_args()

    collection = args.collection

    # Find markdown files
    if args.file:
        md_files = sorted(MD_DIR.glob(f"{args.file}.md"))
    else:
        md_files = sorted(MD_DIR.glob("*.md"))

    if not md_files:
        print("No markdown files found!")
        return

    print("=" * 60)
    print("LLM-GUIDED SEMANTIC CHUNKING")
    print(f"  Files: {len(md_files)}")
    print(f"  Collection: {collection}")
    print(f"  Hybrid: Dense (text-embedding-3-large) + Sparse (BM25)")
    print(f"  LLM: Azure OpenAI / {LLM_DEPLOYMENT}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 60)

    qdrant_client = QdrantClient(
        url=qdrant_ingest.QDRANT_URL, api_key=qdrant_ingest.QDRANT_API_KEY
    )

    if not args.dry_run:
        dim = qdrant_ingest.infer_embedding_dim()

        if args.clear:
            try:
                qdrant_client.delete_collection(collection)
                print(f"[qdrant] cleared collection: {collection}")
            except Exception:
                pass

        qdrant_ingest.ensure_collection(qdrant_client, collection, dim)

    total = len(md_files)
    args_list = [
        (md, qdrant_client, collection, i + 1, total, args.dry_run)
        for i, md in enumerate(md_files)
    ]

    total_chunks = 0
    max_workers = 8  # Safe with ~3000 req/min GPT-4o + ~978K tok/min embeddings
    print(f"  Workers: {max_workers} parallel")

    failed_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(ingest_single_md_llm, arg): arg[0]
            for arg in args_list
        }
        for future in as_completed(futures):
            md_file = futures[future]
            try:
                total_chunks += future.result()
            except Exception as e:
                print(f"[BATCH] FAILED {md_file.name}: {e}")
                failed_files.append(md_file.name)

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print(f"DRY RUN complete — previewed {total} files")
    else:
        print(f"DONE — ingested {total_chunks} chunks from {total} files → {collection}")
    if failed_files:
        print(f"FAILED ({len(failed_files)}): {', '.join(failed_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
