"""
Gemini 2.5 Flash Document Understanding — PDF → Markdown converter.
Replaces both Docling and separate vision calls with a single API per page batch.

Features:
  - Native PDF understanding (no OCR library needed)
  - Tables extracted as proper markdown
  - Images/screenshots described inline with [IMAGE: ...] tags
  - Parallel page-batch processing for large docs
  - Per-page headers matching pipeline format
"""

import base64
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import oci
from oci.generative_ai_inference.models import (
    ChatDetails,
    DocumentContent,
    DocumentUrl,
    GenericChatRequest,
    OnDemandServingMode,
    TextContent,
    UserMessage,
)

from oci_clients import OCI_COMPARTMENT_ID, get_vision_client

logger = logging.getLogger("gemini_converter")

# Config
BATCH_SIZE = int(os.getenv("GEMINI_DU_BATCH_SIZE", "5"))  # Pages per API call
MAX_WORKERS = int(os.getenv("GEMINI_DU_WORKERS", "2"))     # Parallel API calls
MAX_TOKENS = int(os.getenv("GEMINI_DU_MAX_TOKENS", "8192"))
MODEL = os.getenv("GEMINI_DU_MODEL", "google.gemini-2.5-flash")

EXTRACT_PROMPT = """Extract ALL content from pages {start} to {end} of this PDF as Markdown.
Rules:
- For each page add "# {doc_title} — Page N" header
- ALL text exactly as written — do not summarize or skip anything
- Tables as proper markdown tables with | and ---
- Images/charts/diagrams/screenshots: wrap in <figure> tags with description:
  <figure>
  [IMAGE: detailed description including ALL visible text, numbers, field values from the image]
  </figure>
- Preserve all numbers, dates, names, Arabic text exactly
- Include headers, bullet points, numbered lists as markdown
- Include every form field label and value visible"""

PAGE_SEPARATOR = "\n\n---\n\n"


def _get_page_count(pdf_path: Path) -> Optional[int]:
    """Get PDF page count."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    except ImportError:
        pass
    try:
        from PyPDF2 import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return None


def _extract_batch(
    pdf_b64: str,
    start: int,
    end: int,
    total: int,
    doc_title: str,
    retries: int = 3,
) -> str:
    """Extract a batch of pages from PDF using Gemini 2.5 Flash."""
    client = get_vision_client()
    prompt = EXTRACT_PROMPT.format(
        start=start, end=min(end, total), doc_title=doc_title
    )

    req = GenericChatRequest(
        messages=[
            UserMessage(content=[
                DocumentContent(document_url=DocumentUrl(
                    url=f"data:application/pdf;base64,{pdf_b64}"
                )),
                TextContent(text=prompt),
            ])
        ],
        max_tokens=MAX_TOKENS,
        is_stream=False,
    )
    details = ChatDetails(
        compartment_id=OCI_COMPARTMENT_ID,
        serving_mode=OnDemandServingMode(model_id=MODEL),
        chat_request=req,
    )

    for attempt in range(retries):
        try:
            resp = client.chat(details)
            cr = resp.data.chat_response
            if cr.choices and cr.choices[0].message and cr.choices[0].message.content:
                return cr.choices[0].message.content[0].text
            return ""
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                sleep_s = 15 * (attempt + 1)
                logger.warning(f"Rate limited, sleeping {sleep_s}s...")
                time.sleep(sleep_s)
            elif attempt < retries - 1:
                sleep_s = 5 * (attempt + 1)
                logger.warning(f"Retry {attempt+1}: {e}")
                time.sleep(sleep_s)
            else:
                logger.error(f"Failed pages {start}-{end} after {retries} retries: {e}")
                return f"[ERROR extracting pages {start}-{end}]"


def convert_pdf_to_markdown(
    pdf_path: Path,
    out_dir: Path,
    images_dir: Optional[Path] = None,  # Not used — images described inline
) -> Tuple[Path, Optional[Path]]:
    """
    Convert PDF to Markdown using Gemini 2.5 Flash Document Understanding.

    Processes pages in parallel batches. Each batch is a single API call
    that extracts text, tables, and image descriptions.

    Args:
        pdf_path: Path to PDF file
        out_dir: Directory for output .md file
        images_dir: Ignored (kept for API compatibility with docling_converter)

    Returns:
        (md_path, figures_json_path) — figures_json_path is always None
        (images are described inline in the markdown as [IMAGE: ...])
    """
    doc_id = pdf_path.stem
    doc_title = doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read PDF
    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()

    # Get page count
    total_pages = _get_page_count(pdf_path)
    if total_pages is None:
        total_pages = 20  # Fallback guess
        logger.warning(f"Could not determine page count for {pdf_path.name}, assuming {total_pages}")

    # Build page batches
    batches = []
    for start in range(1, total_pages + 1, BATCH_SIZE):
        end = min(start + BATCH_SIZE - 1, total_pages)
        batches.append((start, end))

    logger.info(f"Converting {pdf_path.name}: {total_pages} pages → {len(batches)} batches ({MAX_WORKERS} parallel)")

    # Process batches in parallel
    t0 = time.time()
    results = {}

    if len(batches) == 1:
        # Single batch — no threading overhead
        s, e = batches[0]
        results[(s, e)] = _extract_batch(pdf_b64, s, e, total_pages, doc_title)
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for s, e in batches:
                future = executor.submit(
                    _extract_batch, pdf_b64, s, e, total_pages, doc_title
                )
                futures[future] = (s, e)

            for future in as_completed(futures):
                s, e = futures[future]
                try:
                    results[(s, e)] = future.result()
                except Exception as ex:
                    logger.error(f"Batch {s}-{e} failed: {ex}")
                    results[(s, e)] = f"[ERROR pages {s}-{e}]"

    # Combine in page order
    full_md = PAGE_SEPARATOR.join(results[k] for k in sorted(results.keys()))

    # Ensure page headers match pipeline format if not already
    # The prompt asks for "# DocTitle — Page N" format
    # Add source file line if missing
    if f"> Source file:" not in full_md:
        # Add source lines after each page header
        def add_source(match):
            page_header = match.group(0)
            page_num = match.group(1) if match.lastindex else "?"
            return f"{page_header}\n\n> Source file: `{pdf_path.name}` • Page {page_num}\n"

        full_md = re.sub(
            rf"# {re.escape(doc_title)} — Page (\d+)",
            add_source,
            full_md,
        )

    elapsed = time.time() - t0
    md_path = out_dir / f"{doc_id}.md"
    md_path.write_text(full_md, encoding="utf-8")

    logger.info(
        f"Converted {pdf_path.name}: {len(full_md):,} chars, "
        f"{total_pages} pages in {elapsed:.1f}s"
    )

    # No separate figures JSON — images described inline
    return md_path, None
