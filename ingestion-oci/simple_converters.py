"""
Non-PDF source converters for the OCI ingestion pipeline.

PDFs go through Gemini Document Understanding (gemini_converter). DOCX and plain
text don't need vision — they're converted to Markdown directly here, then fed
into the SAME downstream chunk → embed → Qdrant path. Each converter writes
`{stem}.md` into `out_dir` and returns its Path, matching process_pdf_oci's
output contract (minus the figures sidecar, which text formats don't have).
"""
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn


def _iter_block_items(doc):
    """Yield paragraphs and tables in document order (python-docx exposes them
    on separate collections; walking the body XML preserves their interleaving
    so a table stays next to the heading it belongs under)."""
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _table_to_md(table) -> str:
    """Render a docx table as a GitHub-flavored Markdown table. The downstream
    typed chunker treats Markdown tables as table chunks and embeds their values,
    so this keeps table cells searchable (same as the PDF table path)."""
    rows = []
    for r in table.rows:
        cells = [(c.text or "").replace("\n", " ").replace("|", "\\|").strip() for c in r.cells]
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    ncol = len(table.rows[0].cells)
    sep = "| " + " | ".join(["---"] * ncol) + " |"
    return rows[0] + "\n" + sep + ("\n" + "\n".join(rows[1:]) if len(rows) > 1 else "")


def docx_to_markdown(src: Path, out_dir: Path) -> Path:
    """Convert a .docx to Markdown: headings → #, paragraphs → text, tables →
    Markdown tables, in document order."""
    src, out_dir = Path(src), Path(out_dir)
    doc = Document(str(src))
    parts = []
    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = ((block.style.name if block.style else "") or "").lower()
            if style == "title":
                parts.append(f"# {text}")
            elif style.startswith("heading"):
                lvl = "".join(ch for ch in style if ch.isdigit())
                parts.append(f"{'#' * (int(lvl) if lvl else 2)} {text}")
            else:
                parts.append(text)
        elif isinstance(block, Table):
            md = _table_to_md(block)
            if md:
                parts.append(md)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{src.stem}.md"
    md_path.write_text("\n\n".join(parts).rstrip() + "\n", encoding="utf-8")
    return md_path


def text_to_markdown(src: Path, out_dir: Path) -> Path:
    """Pass plain text / Markdown straight through to a `{stem}.md` file —
    no conversion needed; the chunker reads Markdown natively."""
    src, out_dir = Path(src), Path(out_dir)
    raw = src.read_text(encoding="utf-8", errors="ignore")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{src.stem}.md"
    md_path.write_text(raw if raw.endswith("\n") else raw + "\n", encoding="utf-8")
    return md_path
