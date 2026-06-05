#!/usr/bin/env python3
"""
Chunk type definitions and builders for multimodal RAG.
Defines the structure of different chunk types and provides builders.
"""

import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class ChunkType(str, Enum):
    """Types of chunks in the multimodal RAG system."""
    IMAGE_DESCRIPTION = "image_description"  # From JSON figure descriptions
    OCR_DETAIL = "ocr_detail"                # From <figure> tag content
    PAGE_CONTEXT = "page_context"            # Text outside figures
    CONTROL = "control"                      # Control statements
    DEFINITION = "definition"                # Notes/definitions
    DOC_SUMMARY = "doc_summary"              # Document-level summary
    TABLE_SUMMARY = "table_summary"          # Table summaries (not full tables)


class FigureType(str, Enum):
    """Types of figures/images."""
    UI_SCREEN = "ui_screen"
    FLOWCHART = "flowchart"
    MIXED = "mixed"
    LOGO = "logo"
    OTHER = "other"


@dataclass
class BaseChunk:
    """Base class for all chunk types."""
    # Identity
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    chunk_type: str = ""
    chunk_index: int = 0
    
    # Document context
    source_file: str = ""
    doc_id: str = ""
    domain: str = ""              # ABS, ACC, HRD, etc.
    function: str = ""            # SPD, CNA, DMD, etc.
    variant: str = ""             # W, G, P, B, F, R
    page: int = 1
    
    # Content
    text: str = ""
    
    # Flags
    multimodal: bool = True
    is_primary_content: bool = True
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload dict."""
        return {k: v for k, v in asdict(self).items() if v is not None and v != "" and v != []}


@dataclass
class ImageDescriptionChunk(BaseChunk):
    """Chunk for JSON figure descriptions (primary image content)."""
    chunk_type: str = ChunkType.IMAGE_DESCRIPTION.value
    
    # Figure metadata
    figure_id: str = ""
    figure_type: str = FigureType.OTHER.value
    image_path: str = ""
    caption: str = ""
    
    # Extracted entities
    roles: List[str] = field(default_factory=list)
    decision_points: List[str] = field(default_factory=list)
    has_steps: bool = False
    
    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        # Ensure lists are included even if empty for filtering
        payload['roles'] = self.roles
        payload['decision_points'] = self.decision_points
        return payload


@dataclass
class OCRDetailChunk(BaseChunk):
    """Chunk for OCR-extracted text from <figure> tags (supplementary)."""
    chunk_type: str = ChunkType.OCR_DETAIL.value
    is_primary_content: bool = False
    
    # Figure reference
    figure_idx: int = 0
    
    # Extracted entities
    roles: List[str] = field(default_factory=list)
    decision_points: List[str] = field(default_factory=list)
    has_steps: bool = False
    
    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload['roles'] = self.roles
        payload['decision_points'] = self.decision_points
        return payload


@dataclass
class PageContextChunk(BaseChunk):
    """Chunk for text content outside figures on a page."""
    chunk_type: str = ChunkType.PAGE_CONTEXT.value
    
    # Page info
    has_controls: bool = False
    has_notes: bool = False
    has_tables: bool = False


@dataclass
class ControlChunk(BaseChunk):
    """Chunk for individual control statements."""
    chunk_type: str = ChunkType.CONTROL.value
    
    # Control info
    control_number: int = 0


@dataclass
class DefinitionChunk(BaseChunk):
    """Chunk for notes and definitions."""
    chunk_type: str = ChunkType.DEFINITION.value
    
    # Note info
    note_id: str = ""
    definition_terms: List[str] = field(default_factory=list)
    
    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload['definition_terms'] = self.definition_terms
        return payload


@dataclass
class DocSummaryChunk(BaseChunk):
    """Chunk for document-level summary."""
    chunk_type: str = ChunkType.DOC_SUMMARY.value
    
    # Summary metadata
    total_pages: int = 1
    figure_types: List[str] = field(default_factory=list)
    key_roles: List[str] = field(default_factory=list)
    control_count: int = 0
    
    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload['figure_types'] = self.figure_types
        payload['key_roles'] = self.key_roles
        return payload


@dataclass
class TableSummaryChunk(BaseChunk):
    """Chunk for table summaries (embed summary, store full table in metadata)."""
    chunk_type: str = ChunkType.TABLE_SUMMARY.value
    
    # Table metadata
    table_idx: int = 0
    row_count: int = 0
    column_count: int = 0
    header: str = ""  # Table header row for context
    full_table: str = ""  # Complete table content (stored in metadata, not embedded)


# =============================================================================
# CHUNK BUILDERS
# =============================================================================

class ChunkBuilder:
    """Builder class for creating chunks with common metadata."""
    
    def __init__(
        self,
        source_file: str,
        doc_id: str,
        domain: str = "",
        function: str = "",
        variant: str = ""
    ):
        self.source_file = source_file
        self.doc_id = doc_id
        self.domain = domain
        self.function = function
        self.variant = variant
        self._chunk_index = 0
    
    def _next_index(self) -> int:
        idx = self._chunk_index
        self._chunk_index += 1
        return idx
    
    def _base_kwargs(self, page: int = 1) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "doc_id": self.doc_id,
            "domain": self.domain,
            "function": self.function,
            "variant": self.variant,
            "page": page,
            "chunk_index": self._next_index(),
        }
    
    def build_image_description(
        self,
        text: str,
        page: int,
        figure_id: str,
        figure_type: str,
        image_path: str = "",
        caption: str = "",
        roles: List[str] = None,
        decision_points: List[str] = None,
        has_steps: bool = False
    ) -> ImageDescriptionChunk:
        """Build an image description chunk."""
        return ImageDescriptionChunk(
            **self._base_kwargs(page),
            text=text,
            figure_id=figure_id,
            figure_type=figure_type,
            image_path=image_path,
            caption=caption,
            roles=roles or [],
            decision_points=decision_points or [],
            has_steps=has_steps,
        )
    
    def build_ocr_detail(
        self,
        text: str,
        page: int,
        figure_idx: int,
        roles: List[str] = None,
        decision_points: List[str] = None,
        has_steps: bool = False
    ) -> OCRDetailChunk:
        """Build an OCR detail chunk."""
        return OCRDetailChunk(
            **self._base_kwargs(page),
            text=text,
            figure_idx=figure_idx,
            roles=roles or [],
            decision_points=decision_points or [],
            has_steps=has_steps,
        )
    
    def build_page_context(
        self,
        text: str,
        page: int,
        has_controls: bool = False,
        has_notes: bool = False,
        has_tables: bool = False
    ) -> PageContextChunk:
        """Build a page context chunk."""
        return PageContextChunk(
            **self._base_kwargs(page),
            text=text,
            has_controls=has_controls,
            has_notes=has_notes,
            has_tables=has_tables,
        )
    
    def build_control(
        self,
        text: str,
        page: int,
        control_number: int
    ) -> ControlChunk:
        """Build a control chunk."""
        return ControlChunk(
            **self._base_kwargs(page),
            text=text,
            control_number=control_number,
        )
    
    def build_definition(
        self,
        text: str,
        page: int,
        note_id: str,
        definition_terms: List[str] = None
    ) -> DefinitionChunk:
        """Build a definition chunk."""
        return DefinitionChunk(
            **self._base_kwargs(page),
            text=text,
            note_id=note_id,
            definition_terms=definition_terms or [],
        )
    
    def build_doc_summary(
        self,
        text: str,
        total_pages: int,
        figure_types: List[str] = None,
        key_roles: List[str] = None,
        control_count: int = 0
    ) -> DocSummaryChunk:
        """Build a document summary chunk."""
        return DocSummaryChunk(
            **self._base_kwargs(1),
            text=text,
            total_pages=total_pages,
            figure_types=figure_types or [],
            key_roles=key_roles or [],
            control_count=control_count,
        )
    
    def build_table_summary(
        self,
        text: str,
        page: int,
        table_idx: int,
        row_count: int = 0,
        column_count: int = 0,
        header: str = "",
        full_table: str = ""
    ) -> TableSummaryChunk:
        """Build a table summary chunk. Embeds summary, stores full table in metadata."""
        return TableSummaryChunk(
            **self._base_kwargs(page),
            text=text,
            table_idx=table_idx,
            row_count=row_count,
            column_count=column_count,
            header=header,
            full_table=full_table,
        )


# =============================================================================
# CHUNK SIZE CONFIG
# =============================================================================

@dataclass
class ChunkSizeConfig:
    """Configuration for chunk size limits by type."""
    
    # Image description chunks (from JSON)
    image_desc_target: int = 150    # Already optimal from analysis
    image_desc_max: int = 300
    image_desc_min: int = 50
    
    # OCR detail chunks
    ocr_target: int = 500
    ocr_max: int = 800
    ocr_min: int = 50               # Skip if below this
    
    # Page context chunks
    page_context_target: int = 600
    page_context_max: int = 1200
    page_context_min: int = 100
    page_context_overlap: int = 100
    
    # Control chunks
    control_max: int = 150
    control_min: int = 20
    
    # Definition chunks
    definition_max: int = 400
    definition_min: int = 30
    
    # Doc summary chunks
    summary_target: int = 200
    summary_max: int = 300
    summary_min: int = 100


# Default config
DEFAULT_CHUNK_CONFIG = ChunkSizeConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_tokens_rough(text: str) -> int:
    """Rough token count estimate (1 token ~ 4 chars)."""
    return max(1, len(text) // 4)


def should_skip_chunk(text: str, min_tokens: int) -> bool:
    """Check if chunk should be skipped due to insufficient content."""
    return count_tokens_rough(text) < min_tokens


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."


def get_chunk_type_for_query(query: str) -> List[str]:
    """
    Determine which chunk types to prioritize for a query.
    Used for filtered retrieval.
    """
    query_lower = query.lower()
    
    prioritized = []
    
    # UI/screen questions
    if any(k in query_lower for k in ['screen', 'click', 'button', 'where', 'interface', 'show me']):
        prioritized.append(ChunkType.IMAGE_DESCRIPTION.value)
    
    # Process questions
    if any(k in query_lower for k in ['how', 'process', 'workflow', 'step', 'procedure']):
        prioritized.extend([ChunkType.IMAGE_DESCRIPTION.value, ChunkType.OCR_DETAIL.value])
    
    # Control questions
    if any(k in query_lower for k in ['control', 'check', 'validation']):
        prioritized.append(ChunkType.CONTROL.value)
    
    # Definition questions
    if any(k in query_lower for k in ['what is', 'define', 'meaning', 'abbreviation']):
        prioritized.append(ChunkType.DEFINITION.value)
    
    # Role questions
    if any(k in query_lower for k in ['who', 'responsible', 'role', 'team', 'manager']):
        prioritized.extend([ChunkType.IMAGE_DESCRIPTION.value, ChunkType.DOC_SUMMARY.value])
    
    # Document-level questions
    if any(k in query_lower for k in ['document', 'about', 'cover', 'topic']):
        prioritized.append(ChunkType.DOC_SUMMARY.value)
    
    # Default: return all types
    if not prioritized:
        return [ct.value for ct in ChunkType]
    
    return list(dict.fromkeys(prioritized))  # Remove duplicates, preserve order


if __name__ == "__main__":
    # Test the chunk builders
    builder = ChunkBuilder(
        source_file="ABS - SPD - 006 - Import Shipment Freight - W -1.md",
        doc_id="ABS - SPD - 006 - Import Shipment Freight - W -1",
        domain="ABS",
        function="SPD",
        variant="W"
    )
    
    # Build image description chunk
    img_chunk = builder.build_image_description(
        text="This image shows a process flowchart for import shipment...",
        page=1,
        figure_id="1.1",
        figure_type="flowchart",
        image_path="/path/to/image.png",
        roles=["ABS Supply Planning", "Brand Manager"],
        decision_points=["Valid?", "Approved?"],
        has_steps=True
    )
    print("Image Description Chunk:")
    print(img_chunk.to_payload())
    
    # Build control chunk
    ctrl_chunk = builder.build_control(
        text="Control 5: Optimal quotation selected as per target delivery date.",
        page=1,
        control_number=5
    )
    print("\nControl Chunk:")
    print(ctrl_chunk.to_payload())
    
    # Test query routing
    test_queries = [
        "Where do I click to upload a file?",
        "What does Control 5 mean?",
        "What is BL?",
        "How do I create a purchase order?",
        "Who is responsible for approving leave?",
    ]
    
    print("\nQuery Type Routing:")
    for q in test_queries:
        types = get_chunk_type_for_query(q)
        print(f"  '{q[:40]}...' → {types}")
