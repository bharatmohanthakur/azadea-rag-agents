"""
PDF → Markdown via Claude (native Anthropic API) — drop-in alternative to
gemini_converter for the Document-Understanding stage.

Same contract as gemini_converter.convert_pdf_to_markdown:
    convert_pdf_to_markdown(pdf_path, out_dir, images_dir=None) -> (md_path, None)
Same extraction prompt, same page headers ("# {title} — Page N"), same
"> Source file:" lines, same batch-parallel structure.

Differences from the Gemini version:
  - Calls the Claude API (claude-sonnet-4-6 by default) instead of OCI GenAI —
    no OCI tenant 429 throttling on ingestion.
  - Each batch call attaches ONLY its page slice (PyMuPDF sub-document), not the
    whole PDF: Claude bills document input per page, so slicing keeps cost and
    tokens proportional to the batch. Falls back to the full PDF if slicing fails.
"""
import base64
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

# ANTHROPIC_API_KEY lives in the root .env (this file sits in ingestion-oci/).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic

logger = logging.getLogger("claude_converter")

BATCH_SIZE = int(os.getenv("CLAUDE_DU_BATCH_SIZE", "5"))       # pages per API call
MAX_WORKERS = int(os.getenv("CLAUDE_DU_WORKERS", "2"))          # parallel batches
MODEL = os.getenv("CLAUDE_DU_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("CLAUDE_DU_MAX_TOKENS", "16000"))
GROUPING_MODEL = os.getenv("CLAUDE_GROUPING_MODEL", "claude-sonnet-4-6")
# FLOOR for the grouping output budget, not a hard cap: the real budget scales
# with page count (≈60 tokens per possible GROUP line) so large manuals can't
# be silently truncated. max_tokens is a ceiling, not a spend — a generous
# budget costs nothing unless generated.
GROUPING_MAX_TOKENS = int(os.getenv("CLAUDE_GROUPING_MAX_TOKENS", "4000"))
PAGE_SEPARATOR = "\n\n"

# Same instruction set as gemini_converter.EXTRACT_PROMPT, plus one line telling
# the model the attachment is a page slice carrying original page numbers.
EXTRACT_PROMPT = """The attached PDF contains pages {start} to {end} of the document "{doc_title}".
Extract ALL content from these pages as Markdown.
Rules:
- For each page add "# {doc_title} — Page N" header, where N is the ORIGINAL page number ({start}..{end} in order)
- ALL text exactly as written — do not summarize or skip anything
- Tables as proper markdown tables with | and ---
- Images/charts/diagrams/screenshots: wrap in <figure> tags with description:
  <figure>
  [IMAGE: detailed description including ALL visible text, numbers, field values from the image]
  </figure>
- Preserve all numbers, dates, names, Arabic text exactly
- Include headers, bullet points, numbered lists as markdown
- Include every form field label and value visible"""

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def _get_page_count(pdf_path: Path) -> Optional[int]:
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


def _slice_pdf_b64(pdf_path: Path, start: int, end: int) -> str:
    """Base64 of a new PDF containing only pages start..end (1-indexed,
    inclusive). Falls back to the whole file if slicing fails."""
    try:
        import fitz
        src = fitz.open(str(pdf_path))
        sub = fitz.open()
        sub.insert_pdf(src, from_page=start - 1, to_page=end - 1)
        data = sub.tobytes()
        sub.close()
        src.close()
        return base64.standard_b64encode(data).decode()
    except Exception as e:
        logger.warning(f"page-slice failed ({e}); sending whole PDF for batch {start}-{end}")
        return base64.standard_b64encode(pdf_path.read_bytes()).decode()


def _extract_batch(pdf_path: Path, start: int, end: int, total: int,
                   doc_title: str, retries: int = 3) -> str:
    """Extract one page batch via Claude. Retries transient failures with
    backoff; returns an [ERROR ...] marker after exhausting retries (same
    contract as the Gemini converter)."""
    client = _get_client()
    prompt = EXTRACT_PROMPT.format(start=start, end=min(end, total), doc_title=doc_title)
    pdf_b64 = _slice_pdf_b64(pdf_path, start, end)

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "document",
                         "source": {"type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                timeout=300.0,
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        except anthropic.RateLimitError as e:
            sleep_s = 15 * (attempt + 1)
            logger.warning(f"Claude rate limited (pages {start}-{end}), sleeping {sleep_s}s...")
            time.sleep(sleep_s)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < retries - 1:
                sleep_s = 5 * (attempt + 1)
                logger.warning(f"Claude {e.status_code} (pages {start}-{end}), retry in {sleep_s}s")
                time.sleep(sleep_s)
            else:
                logger.error(f"Failed pages {start}-{end}: {e}")
                return f"[ERROR extracting pages {start}-{end}]"
        except Exception as e:
            if attempt < retries - 1:
                sleep_s = 5 * (attempt + 1)
                logger.warning(f"Retry {attempt+1} (pages {start}-{end}): {e}")
                time.sleep(sleep_s)
            else:
                logger.error(f"Failed pages {start}-{end} after {retries} retries: {e}")
                return f"[ERROR extracting pages {start}-{end}]"
    return f"[ERROR extracting pages {start}-{end}]"


def convert_pdf_to_markdown(
    pdf_path: Path,
    out_dir: Path,
    images_dir: Optional[Path] = None,  # unused — images described inline
) -> Tuple[Path, Optional[Path]]:
    """Convert a PDF to Markdown using Claude Document Understanding. Returns
    (md_path, None) — identical contract to gemini_converter."""
    doc_id = pdf_path.stem
    doc_title = doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    total_pages = _get_page_count(pdf_path)
    if total_pages is None:
        total_pages = 20
        logger.warning(f"Could not determine page count for {pdf_path.name}, assuming {total_pages}")

    batches = []
    for start in range(1, total_pages + 1, BATCH_SIZE):
        end = min(start + BATCH_SIZE - 1, total_pages)
        batches.append((start, end))

    logger.info(f"[claude-du] Converting {pdf_path.name}: {total_pages} pages → "
                f"{len(batches)} batches ({MAX_WORKERS} parallel, model={MODEL})")

    t0 = time.time()
    results = {}
    if len(batches) == 1:
        s, e = batches[0]
        results[(s, e)] = _extract_batch(pdf_path, s, e, total_pages, doc_title)
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_extract_batch, pdf_path, s, e, total_pages, doc_title): (s, e)
                       for s, e in batches}
            for future in as_completed(futures):
                s, e = futures[future]
                try:
                    results[(s, e)] = future.result()
                except Exception as ex:
                    logger.error(f"Batch {s}-{e} failed: {ex}")
                    results[(s, e)] = f"[ERROR pages {s}-{e}]"

    full_md = PAGE_SEPARATOR.join(results[k] for k in sorted(results.keys()))

    # Same post-processing as the Gemini converter: ensure source lines exist.
    if "> Source file:" not in full_md:
        def add_source(match):
            page_header = match.group(0)
            page_num = match.group(1) if match.lastindex else "?"
            return f"{page_header}\n\n> Source file: `{pdf_path.name}` • Page {page_num}\n"
        full_md = re.sub(rf"# {re.escape(doc_title)} — Page (\d+)", add_source, full_md)

    elapsed = time.time() - t0
    md_path = out_dir / f"{doc_id}.md"
    md_path.write_text(full_md, encoding="utf-8")
    logger.info(f"[claude-du] Converted {pdf_path.name}: {len(full_md):,} chars, "
                f"{total_pages} pages in {elapsed:.1f}s")
    return md_path, None


# ─────────────────────────────────────────────────────────────────────────────
# Page grouping via Claude — drop-in for oci_pipeline.get_page_groupings_oci
# ─────────────────────────────────────────────────────────────────────────────

def get_page_groupings_claude(doc_text, pages, doc_name):
    """Group pages by topic using Claude (native Anthropic) instead of Gemini on
    OCI GenAI. Same signature and return contract as get_page_groupings_oci:
    (groups, used_fallback). The response parser is reused from oci_pipeline so
    both backends accept the identical GROUP: output format."""
    from oci_pipeline import _parse_grouping_response   # pure parser, no OCI calls

    total_pages = max(pages.keys()) if pages else 1
    if total_pages <= 1:
        return [[1]], False

    page_sections = []
    for pnum in sorted(pages.keys()):
        page_sections.append(f"=== PAGE {pnum} ===\n{pages[pnum].strip()}")
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

    client = _get_client()
    # Output budget scales with the input: worst case is one GROUP line per page.
    budget = max(GROUPING_MAX_TOKENS, 60 * total_pages)
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=GROUPING_MODEL,
                max_tokens=budget,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                timeout=120.0,
            )
            # A truncated group list must NEVER be parsed as truth — pages would
            # silently drop out of groups. Detect and retry with a bigger budget.
            if resp.stop_reason == "max_tokens":
                logger.warning(f"[claude-grouping] output truncated at {budget} tokens "
                               f"for '{doc_name}' (attempt {attempt+1}); doubling budget")
                budget *= 2
                continue
            raw = "".join(b.text for b in resp.content if b.type == "text").strip()
            if raw:
                groups = _parse_grouping_response(raw, total_pages)
                logger.info(f"[claude-grouping] '{doc_name}': {len(groups)} group(s)")
                return groups, False
            logger.warning(f"[claude-grouping] empty response, attempt {attempt+1}")
        except Exception as e:
            sleep_s = 2 ** attempt
            logger.warning(f"[claude-grouping] retry {attempt+1}/3 for '{doc_name}': {e} (sleep {sleep_s}s)")
            time.sleep(sleep_s)

    logger.warning(f"[claude-grouping] FALLBACK: page-by-page for '{doc_name}'")
    return [[p] for p in sorted(pages.keys())], True


# ─────────────────────────────────────────────────────────────────────────────
# Contextual retrieval headers — one line of search vocabulary per chunk
# ─────────────────────────────────────────────────────────────────────────────

def get_section_headers_claude(doc_name, sections):
    """Generate a one-line retrieval-context header for each section of a
    document (one API call for the whole batch). The header names the document,
    its policy area, and — critically — any TERMS the section defines (leave
    types, benefit names, system names), in the vocabulary employees actually
    search with. Prepended to chunks before embedding, it bridges the gap
    between a user's words ("paid off leave type policy rules") and prose that
    never says them.

    sections: list of (heading, text) — text may be truncated by the caller.
    Returns a list of header strings, one per section ('' on failure — callers
    must treat headers as best-effort and never block ingestion on them)."""
    if not sections:
        return []
    numbered = []
    for i, (head, text) in enumerate(sections, 1):
        excerpt = " ".join(text.split())[:600]
        numbered.append(f"[{i}] heading: {head or '(none)'}\n    content: {excerpt}")
    listing = "\n".join(numbered)

    prompt = f"""Document: "{doc_name}" (an internal company policy/procedure document).

For EACH numbered section below, write ONE line of retrieval context (max 28 words) that will be prepended to that section's search chunk. Each line MUST:
- name the document/policy area in plain words (e.g. "Employee Attendance policy")
- say what the section defines or covers
- explicitly NAME any specific terms the section defines — leave types, benefit names, allowance names, system names, form names — in double quotes (e.g. defines the "Paid Off" leave type)
- use the vocabulary an employee would type when searching (policy, rules, leave type, entitlement, how to, eligibility)

Output EXACTLY one line per section, format:
1: <context line>
2: <context line>

{listing}"""

    client = _get_client()
    try:
        resp = client.messages.create(
            model=GROUPING_MODEL,
            max_tokens=max(1000, 60 * len(sections)),
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            timeout=120.0,
        )
        if resp.stop_reason == "max_tokens":
            logger.warning(f"[context-headers] truncated for '{doc_name}'; skipping headers")
            return ["" for _ in sections]
        raw = "".join(b.text for b in resp.content if b.type == "text")
        headers = ["" for _ in sections]
        for line in raw.splitlines():
            m = re.match(r"^\s*(\d+)\s*[:.)-]\s*(.+)$", line.strip())
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(headers):
                    headers[idx] = m.group(2).strip()
        got = sum(1 for h in headers if h)
        logger.info(f"[context-headers] '{doc_name}': {got}/{len(sections)} headers")
        return headers
    except Exception as e:
        logger.warning(f"[context-headers] failed for '{doc_name}' (non-fatal): {e}")
        return ["" for _ in sections]
