"""
Docling-based PDF → Markdown + Figure extraction.
Replaces Azure Document Intelligence entirely.

Output:
  - Markdown file with per-page headers (matching existing pipeline format)
  - _figures.json sidecar with extracted figure images + metadata
"""

import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger("docling_converter")

# Page separator matching existing Azure DI pipeline format
PAGE_SEPARATOR = "\n\n---\n\n"


def _get_converter() -> DocumentConverter:
    """Create Docling converter with image extraction enabled."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    return DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )


def convert_pdf_to_markdown(
    pdf_path: Path,
    out_dir: Path,
    images_dir: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Convert a PDF to Markdown + figures JSON using Docling.

    Args:
        pdf_path: Path to the PDF file
        out_dir: Directory for the output .md file
        images_dir: Directory for extracted figure images (default: out_dir/../images_oci)

    Returns:
        (md_path, figures_json_path) — figures_json_path is None if no figures found
    """
    doc_id = pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if images_dir is None:
        images_dir = out_dir.parent / "images_oci"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting '{pdf_path.name}' with Docling...")

    converter = _get_converter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    # --- Build per-page Markdown (matching Azure DI pipeline format) ---
    # Docling gives us doc.pages with page numbers.
    # We export full markdown and also track page boundaries.
    full_md = doc.export_to_markdown()

    # Build per-page blocks with headers matching existing format:
    # "# DocTitle — Page N\n\n> Source file: `filename.pdf` • Page N\n\n"
    page_blocks = []
    page_texts = _split_markdown_by_pages(full_md, doc)

    for pnum, page_text in sorted(page_texts.items()):
        header = (
            f"# {doc_id} — Page {pnum}\n\n"
            f"> Source file: `{pdf_path.name}` • Page {pnum}\n\n"
        )
        text = page_text.strip() or "_(No text recognized on this page)_"
        page_blocks.append(header + text + "\n")

    if not page_blocks:
        # Fallback: treat entire markdown as one page
        header = f"# {doc_id} — Page 1\n\n> Source file: `{pdf_path.name}` • Page 1\n\n"
        page_blocks.append(header + full_md.strip() + "\n")

    per_doc_markdown = PAGE_SEPARATOR.join(page_blocks).rstrip() + "\n"
    md_path = out_dir / f"{doc_id}.md"
    md_path.write_text(per_doc_markdown, encoding="utf-8")
    logger.info(f"Wrote {md_path.name} ({len(per_doc_markdown):,} chars, {len(page_blocks)} pages)")

    # --- Extract figures ---
    figures_json_path = None
    figures = _extract_figures(doc, doc_id, images_dir)
    if figures:
        figures_json_path = out_dir / f"{doc_id}_figures.json"
        figures_json_path.write_text(json.dumps(figures, indent=2), encoding="utf-8")
        logger.info(f"Extracted {len(figures)} figures → {figures_json_path.name}")

    return md_path, figures_json_path


def _split_markdown_by_pages(full_md: str, doc) -> Dict[int, str]:
    """
    Split Docling markdown into per-page text.
    Uses document element provenance to map content to pages.
    """
    page_texts: Dict[int, List[str]] = {}

    # Try to use body items with provenance
    if hasattr(doc, "body") and doc.body and hasattr(doc.body, "children"):
        for item in doc.body.children:
            prov = item.prov[0] if hasattr(item, "prov") and item.prov else None
            page_no = prov.page_no if prov and hasattr(prov, "page_no") else 1

            text = ""
            if hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(doc=doc)
                except Exception:
                    text = getattr(item, "text", "")
            elif hasattr(item, "text"):
                text = item.text

            if text and text.strip():
                page_texts.setdefault(page_no, []).append(text.strip())

    # If provenance-based split worked
    if page_texts:
        return {pnum: "\n\n".join(texts) for pnum, texts in page_texts.items()}

    # Fallback: one page per doc.pages entry, use full markdown
    num_pages = len(doc.pages) if hasattr(doc, "pages") else 1
    if num_pages <= 1:
        return {1: full_md}

    # Simple split by approximate equal parts
    lines = full_md.split("\n")
    per_page = max(1, len(lines) // num_pages)
    result = {}
    for i in range(num_pages):
        start = i * per_page
        end = start + per_page if i < num_pages - 1 else len(lines)
        result[i + 1] = "\n".join(lines[start:end])
    return result


def _extract_figures(doc, doc_id: str, images_dir: Path) -> List[Dict]:
    """
    Extract figure images from Docling document.

    Returns list of dicts matching the existing _figures.json format:
      [{"id", "caption", "page", "image_path", "image_b64", "index", "description"}]
    """
    if not hasattr(doc, "pictures") or not doc.pictures:
        return []

    figures = []
    for idx, pic in enumerate(doc.pictures, start=1):
        img_ref = getattr(pic, "image", None)
        if img_ref is None:
            continue

        # Get page number
        prov = pic.prov[0] if hasattr(pic, "prov") and pic.prov else None
        page_no = prov.page_no if prov and hasattr(prov, "page_no") else 1

        # Get caption — caption_text may be a method(doc) or property depending on Docling version
        cap_attr = getattr(pic, "caption_text", "")
        if callable(cap_attr):
            try:
                caption = cap_attr(doc) or ""
            except Exception:
                caption = ""
        else:
            caption = cap_attr or ""

        # Get image as bytes
        image_bytes = None
        image_b64 = None

        # Try PIL image
        pil_img = getattr(img_ref, "pil_image", None)
        if pil_img is not None:
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        elif hasattr(img_ref, "uri") and str(img_ref.uri).startswith("data:image"):
            # Extract base64 from data URI
            uri = str(img_ref.uri)
            if ";base64," in uri:
                image_b64 = uri.split(";base64,", 1)[1]
                image_bytes = base64.b64decode(image_b64)

        if image_bytes is None:
            logger.warning(f"Figure {idx}: no image data available, skipping")
            continue

        # Filter small images (likely logos/decorative)
        if len(image_bytes) < 5000:
            logger.debug(f"Figure {idx}: too small ({len(image_bytes)} bytes), skipping")
            continue

        # Save to disk
        safe_doc_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in doc_id)[:50]
        fig_filename = f"{safe_doc_id}_fig{idx}.png"
        fig_path = images_dir / fig_filename
        fig_path.write_bytes(image_bytes)

        figures.append({
            "id": f"figure-{idx}",
            "caption": caption,
            "page": page_no,
            "image_path": str(fig_path),
            "image_b64": image_b64,
            "index": idx,
            "description": "",  # Filled later by OCI vision
        })

    return figures
