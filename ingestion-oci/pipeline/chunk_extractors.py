#!/usr/bin/env python3
"""
Entity and metadata extractors for multimodal RAG chunking.
Extracts roles, decision points, controls, notes, and document metadata.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DocumentMetadata:
    """Parsed document metadata from filename."""
    domain: str = ""           # ABS, ACC, HRD, etc.
    function: str = ""         # SPD, CNA, DMD, etc.
    doc_num: str = ""          # 001, 002, etc.
    title: str = ""            # Document title
    variant: str = ""          # W=workflow, G=guide, P=policy, B=business, F=form, R=reference
    version: str = ""          # Version number
    raw_filename: str = ""


@dataclass
class ControlItem:
    """A single control statement."""
    number: int
    text: str
    page: Optional[int] = None


@dataclass
class NoteItem:
    """A single note/definition."""
    note_id: str
    text: str
    definition_terms: List[str] = field(default_factory=list)
    page: Optional[int] = None


@dataclass
class FigureMetadata:
    """Metadata for a figure/image."""
    figure_id: str
    page: int
    description: str
    caption: str = ""
    image_path: str = ""
    figure_type: str = "other"  # ui_screen, flowchart, logo, mixed, other
    roles: List[str] = field(default_factory=list)
    decision_points: List[str] = field(default_factory=list)
    has_steps: bool = False


# =============================================================================
# FILENAME PARSER
# =============================================================================

def parse_doc_filename(filename: str) -> DocumentMetadata:
    """
    Parse document filename into structured metadata.
    
    Examples:
        'ABS - SPD - 006 - Import Shipment Freight - W -1.md'
        'ACC - NONM - 003 - Asset Automation - G - 1.md'
        'HRD - TMD - 001 - Disciplinary Action - W - 2.md'
    
    Returns:
        DocumentMetadata with parsed fields
    """
    meta = DocumentMetadata(raw_filename=filename)
    
    # Remove extension
    name = re.sub(r'\.(md|json)$', '', filename, flags=re.IGNORECASE)
    
    # Pattern: DOMAIN - FUNCTION - NUM - TITLE - VARIANT - VERSION
    # Some files have extra spaces or variations
    pattern = r'^([A-Z]+)\s*-\s*([A-Z]+)\s*-\s*(\d+)\s*-\s*(.+?)\s*-\s*([A-Z])\s*-?\s*(\d+)?$'
    
    match = re.match(pattern, name.strip())
    if match:
        meta.domain = match.group(1).strip()
        meta.function = match.group(2).strip()
        meta.doc_num = match.group(3).strip()
        meta.title = match.group(4).strip()
        meta.variant = match.group(5).strip()
        meta.version = match.group(6).strip() if match.group(6) else "1"
    else:
        # Fallback: try simpler patterns
        parts = name.split(' - ')
        if len(parts) >= 2:
            meta.domain = parts[0].strip() if parts[0].strip().isupper() else ""
            meta.function = parts[1].strip() if len(parts) > 1 and parts[1].strip().isupper() else ""
            meta.title = ' - '.join(parts[2:]) if len(parts) > 2 else name
    
    return meta


def get_variant_description(variant: str) -> str:
    """Get human-readable description for variant code."""
    variants = {
        'W': 'Workflow',
        'G': 'Guide',
        'P': 'Policy',
        'B': 'Business Document',
        'F': 'Form',
        'R': 'Reference',
        'A': 'Appendix',
    }
    return variants.get(variant.upper(), 'Document')


# =============================================================================
# FIGURE TYPE CLASSIFIER
# =============================================================================

# Keywords for classification
UI_KEYWORDS = [
    'screen', 'interface', 'button', 'menu', 'table', 'grid', 'field', 'form',
    'dialog', 'window', 'click', 'select', 'input', 'dropdown', 'checkbox',
    'tab', 'panel', 'toolbar', 'icon', 'column', 'row', 'cell', 'header',
    'footer', 'popup', 'modal', 'screenshot', 'software', 'application',
    'navigation', 'sidebar', 'displays', 'shows a', 'interface from'
]

PROCESS_KEYWORDS = [
    'flowchart', 'process', 'decision', 'step', 'arrow', 'flow', 'diagram',
    'workflow', 'sequence', 'procedure', 'activity', 'swimlane', 'swim lane',
    'start', 'end node', 'branch', 'merge', 'parallel', 'gateway', 'path',
    'process flow', 'detailing the', 'outlining the steps', 'begins with'
]

LOGO_KEYWORDS = [
    'logo', 'microsoft information protection', 'branding', 'cover page',
    'title page', 'company logo', 'azadea logo', 'blue background'
]


def classify_figure_type(description: str) -> str:
    """
    Classify figure type based on description content.
    
    Returns: 'ui_screen', 'flowchart', 'logo', 'mixed', or 'other'
    """
    if not description:
        return 'other'
    
    desc_lower = description.lower()
    
    has_ui = any(k in desc_lower for k in UI_KEYWORDS)
    has_process = any(k in desc_lower for k in PROCESS_KEYWORDS)
    has_logo = any(k in desc_lower for k in LOGO_KEYWORDS)
    
    if has_logo:
        return 'logo'
    if has_process and has_ui:
        return 'mixed'
    if has_process:
        return 'flowchart'
    if has_ui:
        return 'ui_screen'
    return 'other'


# =============================================================================
# ROLE EXTRACTOR
# =============================================================================

ROLE_PATTERNS = [
    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Team)',           # "Supply Planning Team"
    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Manager)',        # "Brand Manager"
    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Officer)',        # "Stock Management Officer"
    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Department)',     # "Legal Department"
    r'(Regional\s+[A-Za-z\s]+)',                          # "Regional Treasury"
    r'(Country\s+[A-Za-z\s]+)',                           # "Country Operations"
    r'(Group\s+[A-Za-z\s]+)',                             # "Group Treasury"
    r'\b(Supplier|Forwarder|Broker|Employee|Requester)\b',  # Single-word roles
    r'(ABS\s+[A-Za-z\s&]+)',                              # "ABS Supply Planning"
    r'(F&[AB]\s+[A-Za-z\s]+)',                            # "F&A Stock Management"
]


def extract_roles(text: str) -> List[str]:
    """
    Extract organizational roles from text.
    
    Returns: List of unique role names found
    """
    if not text:
        return []
    
    roles = set()
    for pattern in ROLE_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            role = match.strip()
            # Clean up role name
            role = re.sub(r'\s+', ' ', role)
            # Filter out too short or too long
            if 3 <= len(role) <= 50:
                roles.add(role)
    
    # Remove substrings (e.g., if we have "ABS Supply Planning Team" and "Supply Planning Team")
    roles_list = list(roles)
    filtered = []
    for r in sorted(roles_list, key=len, reverse=True):
        if not any(r in other and r != other for other in filtered):
            filtered.append(r)
    
    return filtered[:10]  # Limit to 10 roles


# =============================================================================
# DECISION POINT EXTRACTOR
# =============================================================================

def extract_decision_points(text: str) -> List[str]:
    """
    Extract decision points (questions) from text.
    Typically these are Yes/No branching points in flowcharts.
    
    Returns: List of decision point questions
    """
    if not text:
        return []
    
    # Find question patterns
    questions = re.findall(r'([A-Z][^.!?]{5,60}\?)', text)
    
    # Also look for common decision patterns without ?
    decision_patterns = [
        r'\b(Valid)\b',
        r'\b(Approved)\b',
        r'\b(Required)\b',
        r'\b(Needed)\b',
        r'\b(Acceptable)\b',
    ]
    
    decisions = list(set(questions))
    
    # Clean and filter
    cleaned = []
    for d in decisions:
        d = d.strip()
        # Skip if too generic
        if d.lower() in ['what?', 'how?', 'why?', 'when?', 'where?']:
            continue
        # Skip URL fragments
        if 'http' in d.lower() or '.com' in d.lower():
            continue
        cleaned.append(d)
    
    return cleaned[:10]  # Limit to 10


# =============================================================================
# CONTROL EXTRACTOR
# =============================================================================

def extract_controls(text: str) -> List[ControlItem]:
    """
    Extract control statements from text.
    
    Pattern: "Control 1: Description of the control..."
    
    Returns: List of ControlItem objects
    """
    if not text:
        return []
    
    controls = []
    
    # Pattern 1: "Control N: text"
    pattern1 = r'Control\s+(\d+)\s*:\s*(.+?)(?=Control\s+\d+\s*:|# Notes:|$)'
    matches = re.findall(pattern1, text, re.DOTALL | re.IGNORECASE)
    
    for num, desc in matches:
        desc_clean = desc.strip()
        desc_clean = re.sub(r'\s+', ' ', desc_clean)
        if desc_clean:
            controls.append(ControlItem(
                number=int(num),
                text=f"Control {num}: {desc_clean}"
            ))
    
    # Pattern 2: "Control N. text" (alternate format)
    if not controls:
        pattern2 = r'Control\s+(\d+)\.\s*(.+?)(?=Control\s+\d+\.|# Notes:|$)'
        matches = re.findall(pattern2, text, re.DOTALL | re.IGNORECASE)
        for num, desc in matches:
            desc_clean = desc.strip()
            desc_clean = re.sub(r'\s+', ' ', desc_clean)
            if desc_clean:
                controls.append(ControlItem(
                    number=int(num),
                    text=f"Control {num}: {desc_clean}"
                ))
    
    return controls


# =============================================================================
# NOTES/DEFINITION EXTRACTOR
# =============================================================================

# Common abbreviation patterns
ABBREVIATION_PATTERN = r'\b([A-Z]{2,5})\s*:\s*([A-Za-z][A-Za-z\s]+)'


def extract_notes(text: str) -> List[NoteItem]:
    """
    Extract notes and definitions from text.
    
    Patterns:
        "(1) Requester: Brand Manager, Procurement Team..."
        "(6) BL: Bill of Lading"
    
    Returns: List of NoteItem objects
    """
    if not text:
        return []
    
    notes = []
    
    # ONLY extract notes from a dedicated Notes section (not entire document)
    notes_match = re.search(r'#\s*Notes\s*:?\s*\n(.*?)(?=\n#|\Z)', text, re.DOTALL | re.IGNORECASE)
    if not notes_match:
        # No Notes section found - don't try to extract from entire document
        return []
    
    notes_text = notes_match.group(1)
    
    # Pattern: (N) text - only match if followed by actual text content (not just numbers/symbols)
    # This prevents matching table cells like (853) | |
    pattern = r'\((\d{1,2})\)\s*([A-Za-z][^()]{10,500}?)(?=\(\d{1,2}\)|$)'
    matches = re.findall(pattern, notes_text, re.DOTALL)
    
    for note_id, content in matches:
        content_clean = content.strip()
        content_clean = re.sub(r'\s+', ' ', content_clean)
        
        if not content_clean:
            continue
        
        # Extract any abbreviation definitions
        terms = []
        abbrev_matches = re.findall(ABBREVIATION_PATTERN, content_clean)
        for abbrev, definition in abbrev_matches:
            terms.append(abbrev)
        
        notes.append(NoteItem(
            note_id=note_id,
            text=f"({note_id}) {content_clean}",
            definition_terms=terms
        ))
    
    return notes


def extract_abbreviation_definitions(text: str) -> Dict[str, str]:
    """
    Extract abbreviation definitions from text.
    
    Returns: Dict mapping abbreviation to definition
    """
    definitions = {}
    matches = re.findall(ABBREVIATION_PATTERN, text)
    for abbrev, definition in matches:
        definitions[abbrev] = definition.strip()
    return definitions


# =============================================================================
# PAGE EXTRACTOR
# =============================================================================

def extract_page_number(text: str) -> Optional[int]:
    """Extract page number from text markers."""
    # Pattern: "— Page N" or "Page N"
    match = re.search(r'—?\s*Page\s+(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def split_by_pages(text: str) -> Dict[int, str]:
    """
    Split document text by page markers.
    
    Returns: Dict mapping page number to page content
    """
    pages = {}
    
    # Split by page markers
    parts = re.split(r'(#[^#\n]*—\s*Page\s+\d+)', text)
    
    current_page = 1
    current_content = []
    
    for part in parts:
        page_match = re.search(r'—\s*Page\s+(\d+)', part)
        if page_match:
            # Save previous page
            if current_content:
                pages[current_page] = '\n'.join(current_content)
            current_page = int(page_match.group(1))
            current_content = [part]
        else:
            current_content.append(part)
    
    # Don't forget the last page
    if current_content:
        pages[current_page] = '\n'.join(current_content)
    
    return pages


# =============================================================================
# TABLE EXTRACTOR
# =============================================================================

def extract_markdown_tables(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract markdown tables from text.
    
    Returns: List of (table_text, start_pos, end_pos) tuples
    """
    tables = []
    # Match markdown table: header row, separator row (|---|), and data rows
    pattern = r'(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)'
    
    for match in re.finditer(pattern, text):
        table_text = match.group(1)
        # Only include tables with at least 3 rows (header + sep + 1 data)
        rows = table_text.strip().split('\n')
        if len(rows) >= 3:
            tables.append((table_text, match.start(), match.end()))
    
    return tables


def count_table_rows(table_text: str) -> int:
    """Count the number of data rows in a markdown table."""
    rows = table_text.strip().split('\n')
    # Subtract header and separator rows
    return max(0, len(rows) - 2)


def get_table_header(table_text: str) -> str:
    """Extract the header row from a markdown table."""
    rows = table_text.strip().split('\n')
    if rows:
        return rows[0]
    return ""


def create_table_summary_prompt(table_text: str, context_before: str, context_after: str) -> str:
    """Create a prompt for LLM to summarize a table."""
    return (
        "Summarize the following table in 2-3 sentences. "
        "Focus on: what data it contains, key columns, and what it's used for.\n\n"
        f"--- CONTEXT BEFORE ---\n{context_before[-1000:]}\n\n"
        f"--- TABLE (first 30 rows) ---\n{truncate_table(table_text, max_rows=30)}\n\n"
        f"--- CONTEXT AFTER ---\n{context_after[:500]}\n\n"
        "Summary:"
    )


def truncate_table(table_text: str, max_rows: int = 30) -> str:
    """Truncate a table to max_rows for summarization."""
    rows = table_text.strip().split('\n')
    if len(rows) <= max_rows + 2:  # header + sep + max_rows
        return table_text
    
    # Keep header, separator, and first max_rows data rows
    truncated = rows[:max_rows + 2]
    truncated.append(f"| ... ({len(rows) - max_rows - 2} more rows) |")
    return '\n'.join(truncated)


# =============================================================================
# FIGURE CONTENT EXTRACTOR
# =============================================================================

def extract_figure_blocks(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract content from figure/image blocks.

    Supports:
      - Azure DI format: <figure>...</figure>
      - Docling format:  <!-- image --> followed by text until next boundary

    Returns: List of (content, start_pos, end_pos) tuples
    """
    figures = []

    # Azure DI format: <figure>...</figure>
    for match in re.finditer(r'<figure>(.*?)</figure>', text, re.DOTALL):
        figures.append((match.group(1), match.start(), match.end()))

    # Docling format: <!-- image --> followed by content until next boundary
    if not figures:
        for match in re.finditer(r'<!-- image -->', text):
            start = match.end()
            # Find next boundary: another image marker, heading, or page separator
            next_bound = re.search(
                r'(?:<!-- image -->|^#{1,3} [^\n]|^---$)',
                text[start:], re.MULTILINE
            )
            end = start + next_bound.start() if next_bound else len(text)
            content = text[start:end].strip()
            if content and len(content) > 20:  # Skip empty image markers
                figures.append((content, match.start(), end))

    return figures


def clean_ocr_text(text: str) -> str:
    """Clean OCR-extracted text from figures."""
    if not text:
        return ""
    
    # Remove common OCR artifacts
    text = re.sub(r'[·•‣⁃]', '', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'[►▶▷→←↑↓]', '', text)
    text = re.sub(r'[₾✓✗×]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def has_meaningful_content(text: str, min_tokens: int = 50) -> bool:
    """Check if text has meaningful content (not just artifacts)."""
    clean = clean_ocr_text(text)
    tokens = len(clean) // 4  # Rough token estimate
    return tokens >= min_tokens


# =============================================================================
# STEP DETECTOR
# =============================================================================

STEP_KEYWORDS = [
    'step', 'validate', 'submit', 'approve', 'review', 'check', 'confirm',
    'send', 'receive', 'create', 'update', 'process', 'notify', 'request',
    'share', 'generate', 'prepare', 'verify', 'complete', 'initiate'
]


def has_steps(text: str) -> bool:
    """Check if text contains step-like content."""
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Check for numbered steps
    if re.search(r'\b\d+\.\s*[A-Z]', text):
        return True
    
    # Check for step keywords
    keyword_count = sum(1 for k in STEP_KEYWORDS if k in text_lower)
    return keyword_count >= 3


# =============================================================================
# DOCUMENT SUMMARY GENERATOR
# =============================================================================

def generate_doc_summary_text(
    doc_meta: DocumentMetadata,
    figures: List[FigureMetadata],
    controls: List[ControlItem],
    total_pages: int
) -> str:
    """
    Generate a summary text for the document.
    
    This creates a searchable summary chunk for routing queries.
    """
    parts = []
    
    # Document identification
    if doc_meta.title:
        parts.append(f"Document: {doc_meta.title}")
    if doc_meta.domain and doc_meta.function:
        parts.append(f"Department: {doc_meta.domain}, Function: {doc_meta.function}")
    if doc_meta.variant:
        variant_desc = get_variant_description(doc_meta.variant)
        parts.append(f"Type: {variant_desc}")
    
    # Content summary
    figure_types = list(set(f.figure_type for f in figures if f.figure_type != 'logo'))
    if figure_types:
        parts.append(f"Contains: {', '.join(figure_types)}")
    
    # Roles mentioned
    all_roles = set()
    for f in figures:
        all_roles.update(f.roles)
    if all_roles:
        parts.append(f"Key roles: {', '.join(list(all_roles)[:5])}")
    
    # Control count
    if controls:
        parts.append(f"Includes {len(controls)} control points")
    
    parts.append(f"Pages: {total_pages}")
    
    return ". ".join(parts)


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_all_metadata(
    md_text: str,
    filename: str,
    figure_json: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Run full extraction pipeline on a document.
    
    Returns dict with:
        - doc_meta: DocumentMetadata
        - figures: List[FigureMetadata]
        - controls: List[ControlItem]
        - notes: List[NoteItem]
        - pages: Dict[int, str]
        - summary: str
    """
    # Parse filename
    doc_meta = parse_doc_filename(filename)
    
    # Split by pages
    pages = split_by_pages(md_text)
    total_pages = max(pages.keys()) if pages else 1
    
    # Extract controls
    controls = extract_controls(md_text)
    
    # Extract notes
    notes = extract_notes(md_text)
    
    # Process figures from JSON
    figures = []
    if figure_json:
        for fig_data in figure_json:
            desc = fig_data.get('description', '')
            fig_type = classify_figure_type(desc)
            
            # Skip logos
            if fig_type == 'logo':
                continue
            
            fig = FigureMetadata(
                figure_id=str(fig_data.get('id', '')),
                page=fig_data.get('page', 1),
                description=desc,
                caption=fig_data.get('caption', ''),
                image_path=fig_data.get('image_path', ''),
                figure_type=fig_type,
                roles=extract_roles(desc),
                decision_points=extract_decision_points(desc),
                has_steps=has_steps(desc)
            )
            figures.append(fig)
    
    # Generate summary
    summary = generate_doc_summary_text(doc_meta, figures, controls, total_pages)
    
    return {
        'doc_meta': doc_meta,
        'figures': figures,
        'controls': controls,
        'notes': notes,
        'pages': pages,
        'total_pages': total_pages,
        'summary': summary,
    }


if __name__ == "__main__":
    # Test the extractors
    test_filename = "ABS - SPD - 006 - Import Shipment Freight - W -1.md"
    meta = parse_doc_filename(test_filename)
    print(f"Parsed: {meta}")
    
    test_text = """
    Control 1: Shipping details and documents cover destination requirements.
    Control 2: HS item codes require registration.
    
    # Notes:
    
    (1) Requester: Brand Manager, Procurement Team
    (6) BL: Bill of Lading
        AWB: Airwaybill
    """
    
    controls = extract_controls(test_text)
    print(f"\nControls: {controls}")
    
    notes = extract_notes(test_text)
    print(f"\nNotes: {notes}")
    
    test_desc = "This image is a process flowchart detailing the workflow for ABS Supply Planning Team and Brand Manager"
    roles = extract_roles(test_desc)
    print(f"\nRoles: {roles}")
    
    fig_type = classify_figure_type(test_desc)
    print(f"\nFigure type: {fig_type}")
