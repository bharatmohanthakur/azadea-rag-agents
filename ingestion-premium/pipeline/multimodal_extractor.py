#!/usr/bin/env python3
"""
Multimodal Document Extractor for Azadea RAG

Extracts figures/images from PDFs using Azure Document Intelligence
and generates descriptions using GPT-4 Vision for multimodal RAG.

Usage:
    from multimodal_extractor import process_document_multimodal
    figures = process_document_multimodal(pdf_path, doc_client, aoai_client)
"""

import os
import base64
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Azure clients
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
    AnalyzeOutputOption,
)
from openai import AzureOpenAI


# ============== CONFIG ==============
VISION_DEPLOYMENT = os.getenv("AOAI_VISION_DEPLOYMENT", "gpt-4.1")
MIN_FIGURE_BYTES = 15000  # Minimum bytes to process (filter tiny logos/icons)
MAX_FIGURES_PER_DOC = 20  # Limit figures per document to control costs
VISION_MAX_TOKENS = 500  # Max tokens for vision description

# Image storage directory
IMAGES_DIR = Path(os.getenv("MULTIMODAL_IMAGES_DIR", "./images"))

# Keywords that indicate a logo/decorative image (skip these)
LOGO_KEYWORDS = ['logo', 'brand', 'icon', 'header', 'footer', 'watermark', 'signature']


def is_likely_logo_or_decorative(image_bytes: bytes, caption: str = "") -> bool:
    """
    Filter out logos, icons, and small decorative images.
    Returns True if image should be SKIPPED.
    """
    # Skip small images (likely logos/icons)
    if len(image_bytes) < MIN_FIGURE_BYTES:
        return True
    
    # Check caption for logo keywords
    caption_lower = caption.lower() if caption else ""
    for keyword in LOGO_KEYWORDS:
        if keyword in caption_lower:
            return True
    
    return False


def save_and_encode_figure(image_bytes: bytes, doc_id: str, fig_id: str) -> Dict[str, str]:
    """
    Save image to disk and return path + base64 for Qdrant storage.
    
    Args:
        image_bytes: Raw image bytes
        doc_id: Document identifier (sanitized filename)
        fig_id: Figure identifier
    
    Returns:
        Dict with image_path and image_b64
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename
    safe_doc_id = "".join(c if c.isalnum() or c in '-_' else '_' for c in doc_id)[:50]
    safe_fig_id = "".join(c if c.isalnum() or c in '-_.' else '_' for c in str(fig_id))
    
    filename = f"{safe_doc_id}_{safe_fig_id}.png"
    filepath = IMAGES_DIR / filename
    filepath.write_bytes(image_bytes)
    
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    return {
        "image_path": str(filepath.absolute()),
        "image_b64": b64
    }


def get_doc_intelligence_client() -> DocumentIntelligenceClient:
    """Get Azure Document Intelligence client."""
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY")
    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))


def get_aoai_client() -> AzureOpenAI:
    """Get Azure OpenAI client for vision."""
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-02-01",
    )


def describe_image_with_gpt4v(
    aoai: AzureOpenAI,
    image_bytes: bytes,
    context: str = "",
    doc_name: str = ""
) -> str:
    """
    Use GPT-4 Vision to describe an image.
    
    Args:
        aoai: Azure OpenAI client
        image_bytes: Raw image bytes
        context: Optional context (e.g., caption from document)
        doc_name: Document name for context
    
    Returns:
        Text description of the image
    """
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    system_prompt = """You are an expert at describing images from HR policy documents and business materials.
Provide detailed, factual descriptions of:
- Charts and graphs (include data points and trends)
- Organizational diagrams (include hierarchy and relationships)
- Process flows and workflows (describe steps)
- Tables (summarize key information)
- Forms and templates (describe structure and fields)

Focus on information that would help answer employee questions about policies and procedures.
Keep descriptions clear and concise (2-4 sentences)."""

    user_content = []
    if context:
        user_content.append({"type": "text", "text": f"Document: {doc_name}\nCaption/Context: {context}\n\nDescribe this image:"})
    else:
        user_content.append({"type": "text", "text": f"Document: {doc_name}\n\nDescribe this image from the document:"})
    
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64_image}"}
    })
    
    for attempt in range(3):
        try:
            response = aoai.chat.completions.create(
                model=VISION_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=VISION_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                sleep_time = 2 ** attempt
                print(f"[vision] retry {attempt+1}/3 after error: {e} (sleep {sleep_time}s)")
                time.sleep(sleep_time)
            else:
                print(f"[vision] failed after 3 attempts: {e}")
                return f"[Image description unavailable: {str(e)[:100]}]"


def extract_figures_from_result(
    client: DocumentIntelligenceClient,
    result: Any,
    operation_id: str,
    doc_name: str = ""
) -> List[Dict[str, Any]]:
    """
    Extract figure images from Document Intelligence analyze result.
    
    Args:
        client: Document Intelligence client
        result: Analyze result object
        operation_id: Operation ID from the poller
        doc_name: Document name for logging
    
    Returns:
        List of figure dictionaries with image_bytes, caption, page, etc.
    """
    figures = []
    
    if not hasattr(result, 'figures') or not result.figures:
        print(f"[figures] No figures detected in {doc_name}")
        return figures
    
    print(f"[figures] Found {len(result.figures)} figures in {doc_name}")
    
    for idx, fig in enumerate(result.figures[:MAX_FIGURES_PER_DOC]):
        try:
            figure_id = fig.id
            
            # Get caption if available
            caption = ""
            if hasattr(fig, 'caption') and fig.caption:
                caption = fig.caption.content if hasattr(fig.caption, 'content') else str(fig.caption)
            
            # Get page number
            page_num = 1
            if hasattr(fig, 'bounding_regions') and fig.bounding_regions:
                page_num = fig.bounding_regions[0].page_number
            
            # Download cropped figure image from API
            # Note: This requires the analyze operation to have been called with output=figures
            try:
                figure_response = client.get_analyze_result_figure(
                    model_id="prebuilt-layout",
                    result_id=operation_id,
                    figure_id=figure_id
                )
                
                # Handle different response types - may be generator, bytes, or response object
                if hasattr(figure_response, 'read'):
                    image_bytes = figure_response.read()
                elif hasattr(figure_response, '__iter__') and not isinstance(figure_response, (bytes, bytearray)):
                    # It's a generator/iterator - join the chunks
                    image_bytes = b''.join(figure_response)
                else:
                    image_bytes = bytes(figure_response)
                
                if not image_bytes:
                    print(f"[figures] Empty image data for figure {figure_id}")
                    continue
                
                # Filter out logos and small decorative images
                if is_likely_logo_or_decorative(image_bytes, caption):
                    print(f"[figures] Skipping figure {idx+1}: likely logo/decorative ({len(image_bytes)} bytes)")
                    continue
                
                # Save image to disk and encode as base64
                image_data = save_and_encode_figure(image_bytes, doc_name, figure_id)
                
                figures.append({
                    "id": figure_id,
                    "caption": caption,
                    "page": page_num,
                    "image_bytes": image_bytes,
                    "image_path": image_data["image_path"],
                    "image_b64": image_data["image_b64"],
                    "index": idx + 1
                })
                print(f"[figures] Extracted figure {idx+1}: page {page_num}, {len(image_bytes)} bytes, saved to {image_data['image_path']}")
                
            except Exception as e:
                print(f"[figures] Failed to download figure {figure_id}: {e}")
                continue
                
        except Exception as e:
            print(f"[figures] Error processing figure {idx}: {e}")
            continue
    
    return figures


def analyze_document_with_figures(
    client: DocumentIntelligenceClient,
    pdf_path: Path,
) -> tuple:
    """
    Analyze document with figure extraction enabled.
    
    Args:
        client: Document Intelligence client
        pdf_path: Path to PDF file
    
    Returns:
        Tuple of (result, operation_id) for figure extraction
    """
    pdf_bytes = pdf_path.read_bytes()
    body = AnalyzeDocumentRequest(bytes_source=pdf_bytes)
    
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=body,
        output_content_format=DocumentContentFormat.MARKDOWN,
        output=[AnalyzeOutputOption.FIGURES],  # Request figure images
    )
    result = poller.result()
    
    # Extract operation ID from poller for figure retrieval
    # The operation_id is typically in the poller details or can be parsed from operation location
    operation_id = None
    if hasattr(poller, 'details') and poller.details:
        operation_id = poller.details.get('operation_id')
    
    if not operation_id:
        # Try to extract from operation location URL
        op_location = getattr(poller, '_operation_location', '') or ''
        if '/analyzeResults/' in op_location:
            operation_id = op_location.split('/analyzeResults/')[-1].split('?')[0]
    
    return result, operation_id


def process_document_multimodal(
    pdf_path: Path,
    doc_client: Optional[DocumentIntelligenceClient] = None,
    aoai_client: Optional[AzureOpenAI] = None,
    existing_result: Any = None,
    operation_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process a document for multimodal content (figures/images).
    
    This is the main entry point for multimodal extraction.
    
    Args:
        pdf_path: Path to the PDF document
        doc_client: Optional Document Intelligence client (created if not provided)
        aoai_client: Optional Azure OpenAI client (created if not provided)
        existing_result: Optional existing analyze result (to avoid re-analyzing)
        operation_id: Optional operation ID if result was already obtained
    
    Returns:
        List of figure dictionaries with descriptions
    """
    doc_name = pdf_path.stem
    
    # Get clients if not provided
    if doc_client is None:
        doc_client = get_doc_intelligence_client()
    if aoai_client is None:
        aoai_client = get_aoai_client()
    
    # Analyze document if no existing result
    if existing_result is None:
        print(f"[multimodal] Analyzing {doc_name} with figure extraction...")
        existing_result, operation_id = analyze_document_with_figures(doc_client, pdf_path)
    
    if not operation_id:
        print(f"[multimodal] Warning: No operation ID available for {doc_name}, cannot extract figures")
        return []
    
    # Extract figures
    figures = extract_figures_from_result(doc_client, existing_result, operation_id, doc_name)
    
    if not figures:
        return []
    
    # Describe each figure with GPT-4 Vision
    print(f"[multimodal] Describing {len(figures)} figures with GPT-4 Vision...")
    for fig in figures:
        description = describe_image_with_gpt4v(
            aoai_client,
            fig['image_bytes'],
            context=fig['caption'],
            doc_name=doc_name
        )
        fig['description'] = description
        # Remove image_bytes from output to save memory
        del fig['image_bytes']
    
    print(f"[multimodal] Completed {doc_name}: {len(figures)} figures described")
    return figures


def format_figures_as_markdown(figures: List[Dict[str, Any]]) -> str:
    """
    Format figure descriptions as markdown for appending to document.
    
    Args:
        figures: List of figure dictionaries with descriptions
    
    Returns:
        Markdown formatted string
    """
    if not figures:
        return ""
    
    md_lines = ["\n\n## Visual Content\n"]
    
    for fig in figures:
        caption = fig.get('caption', f"Figure {fig.get('index', '')}")
        description = fig.get('description', '')
        page = fig.get('page', '')
        
        if caption:
            md_lines.append(f"### {caption}")
        else:
            md_lines.append(f"### Figure (Page {page})")
        
        if page:
            md_lines.append(f"> *Source: Page {page}*\n")
        
        md_lines.append(f"{description}\n")
    
    return "\n".join(md_lines)


# ============== CLI for Testing ==============
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python multimodal_extractor.py <pdf_path>")
        print("Example: python multimodal_extractor.py './HRD - GEN - 001 - Annual Leave - P - 19.pdf'")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Processing: {pdf_path.name}")
    print("=" * 60)
    
    figures = process_document_multimodal(pdf_path)
    
    if figures:
        print("\n" + "=" * 60)
        print("EXTRACTED FIGURES:")
        print("=" * 60)
        for fig in figures:
            print(f"\n--- Figure {fig.get('index', '?')} (Page {fig.get('page', '?')}) ---")
            print(f"Caption: {fig.get('caption', 'N/A')}")
            print(f"Description: {fig.get('description', 'N/A')}")
        
        # Print as markdown
        print("\n" + "=" * 60)
        print("MARKDOWN OUTPUT:")
        print("=" * 60)
        print(format_figures_as_markdown(figures))
    else:
        print("\nNo figures found in document.")
