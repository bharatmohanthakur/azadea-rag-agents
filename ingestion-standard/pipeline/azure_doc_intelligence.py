#!/usr/bin/env python3
"""
Azure Document Intelligence -> Markdown (per-doc + per-page headers) + merged file
- For each PDF in IN_DIR, creates OUT_DIR_PER_DOC/<pdf-stem>.md
- Each page block is prefixed with: "# <DocumentName> — Page <N>"
- (Optional) Also writes a merged ALL_MD that concatenates the per-doc files.
- Fix: converts <table>...</table> blocks to Markdown tables.

Requirements:
    pip install azure-ai-documentintelligence azure-core python-dotenv
"""

import os
import re
import html
from pathlib import Path
from typing import List, Optional, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
    DocumentAnalysisFeature,
)
from dotenv import load_dotenv
load_dotenv()

# -------- CONFIG (edit these) --------
IN_DIR = os.getenv("IN_DIR", ".")           # Folder containing PDFs
OUT_DIR_PER_DOC = os.getenv("OUT_DIR_PER_DOC", "./md_out") # Folder to store one .md per PDF
ALL_MD = "./all_docs.md"               # Merged Markdown across all PDFs (set to None to skip)
LOCALE = None                          # e.g., "en-US"
USE_HIGHRES = True                     # High-resolution OCR feature
PAGE_SEPARATOR = "\n\n---\n\n"         # Between pages inside a doc
DOC_SEPARATOR  = "\n\n# --- MERGE BREAK ---\n\n"  # Between docs in merged file


# ---------- TABLE CONVERTER ----------
_TABLE_RE = re.compile(
    r"<table\b[^>]*>(.*?)</table>",
    re.IGNORECASE | re.DOTALL
)
_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TH_TD_RE = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")  # strip any other tags inside cells

def _clean_cell(text: str) -> str:
    # Unescape HTML entities, strip tags, collapse whitespace, remove pipes/newlines that break MD
    text = html.unescape(text)
    text = _TAG_RE.sub("", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text).strip()
    text = text.replace("\n", " ").replace("|", "\\|")
    return text

def _table_html_to_markdown(table_html: str) -> str:
    # Extract rows
    rows_html = _TR_RE.findall(table_html)
    if not rows_html:
        return table_html  # fallback: leave as-is

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
                # If cell starts with "&", merge with previous
                if c.startswith("&") and merged_cells:
                    merged_cells[-1] += " " + c
                else:
                    merged_cells.append(c)
        
        rows.append(merged_cells)

    # Normalize column counts
    max_cols = max((len(r) for r in rows), default=0)
    if max_cols == 0:
        return table_html

    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    # Header heuristic
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
    """
    Converts any <table>...</table> segments to Markdown tables.
    Leaves non-table content untouched.
    """
    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        try:
            return "\n" + _table_html_to_markdown(inner) + "\n"
        except Exception:
            # on any parsing issue, keep original
            return m.group(0)
    return _TABLE_RE.sub(_repl, markdown_or_html)


# ---------- AZURE CALL + PAGE SPLIT ----------
def analyze_pdf_pages_markdown(
    client: DocumentIntelligenceClient,
    pdf_path: Path,
    locale: Optional[str] = None,
    high_res_ocr: bool = True
) -> Tuple[str, List[str]]:
    """
    Run prebuilt-layout on one PDF.
    Returns:
        (doc_title, per_page_markdowns)
        doc_title is a human header (usually filename stem)
        per_page_markdowns is a list[str], one per page, with tables fixed.
    """
    features = []
    if high_res_ocr:
        features.append(DocumentAnalysisFeature.OCR_HIGH_RESOLUTION)

    pdf_bytes = pdf_path.read_bytes()
    body = AnalyzeDocumentRequest(bytes_source=pdf_bytes)

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",  # try "prebuilt-document" if you need richer structure
        body=body,
        locale=locale,
        output_content_format=DocumentContentFormat.MARKDOWN,  # note: may embed <table> blocks
        features=features,
    )
    result = poller.result()
    full = (result.content or "")

    # If page spans are available, slice per page. Otherwise, single chunk fallback.
    per_page_texts: List[str] = []
    if getattr(result, "pages", None):
        # Each page typically has .spans -> list of (offset, length) segments.
        for p in result.pages:
            page_fragments = []
            spans = getattr(p, "spans", None) or []
            for s in spans:
                start = getattr(s, "offset", 0)
                length = getattr(s, "length", 0)
                if length > 0 and 0 <= start < len(full):
                    page_fragments.append(full[start:start+length])
            page_text = "".join(page_fragments).strip()
            if not page_text:
                # fallback: if spans weird, just skip; we'll handle empty later
                page_text = ""
            # Fix tables
            page_text = convert_xml_tables_to_markdown(page_text)
            if not page_text.endswith("\n"):
                page_text += "\n"
            per_page_texts.append(page_text)
    else:
        # Fallback: whole doc as one page
        chunk = convert_xml_tables_to_markdown(full.strip())
        if not chunk.endswith("\n"):
            chunk += "\n"
        per_page_texts = [chunk]

    return (pdf_path.stem, per_page_texts)


def main():
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
    api_key  = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY") or os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY in environment")

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

    in_dir = Path(IN_DIR).expanduser().resolve()
    out_dir = Path(OUT_DIR_PER_DOC).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs: List[Path] = sorted([p for p in in_dir.rglob("*.pdf") if p.is_file()], key=lambda p: p.name.lower())
    if not pdfs:
        print(f"[WARN] No PDFs found in {in_dir}")
        if ALL_MD:
            Path(ALL_MD).expanduser().resolve().write_text("", encoding="utf-8")
        return

    merged_docs: List[str] = []

    for i, pdf in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] Processing {pdf.name}")
        try:
            doc_title, pages_md = analyze_pdf_pages_markdown(
                client, pdf, locale=LOCALE, high_res_ocr=USE_HIGHRES
            )
        except HttpResponseError as e:
            print(f"[ERROR] Failed on {pdf.name}: {e}")
            continue

        # Build per-doc markdown with page headers
        per_doc_blocks: List[str] = []
        for pnum, page_md in enumerate(pages_md, start=1):
            header = f"# {doc_title} — Page {pnum}\n\n> Source file: `{pdf.name}` • Page {pnum}\n\n"
            # If the page content is empty (rare), keep header so page is recorded
            per_doc_blocks.append(header + (page_md or "_(No text recognized on this page)_\n"))

        per_doc_markdown = PAGE_SEPARATOR.join(per_doc_blocks).rstrip() + "\n"
        # Write per-doc file
        out_path = out_dir / f"{doc_title}.md"
        out_path.write_text(per_doc_markdown, encoding="utf-8")
        print(f"  -> wrote {out_path}")

        if ALL_MD:
            merged_docs.append(per_doc_markdown)

    if ALL_MD:
        merged_all = DOC_SEPARATOR.join(merged_docs).rstrip() + "\n"
        all_path = Path(ALL_MD).expanduser().resolve()
        all_path.parent.mkdir(parents=True, exist_ok=True)
        all_path.write_text(merged_all, encoding="utf-8")
        print(f"[DONE] Wrote merged markdown to {all_path}")
    else:
        print("[DONE] Per-document markdowns written; merged file is disabled.")


if __name__ == "__main__":
    main()
