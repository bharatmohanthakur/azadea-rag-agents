"""
OCI GenAI pipeline functions — replaces all Azure OpenAI calls.

Provides:
  - embed_dense_oci(texts)       → Cohere Embed v4.0 (1536-dim)
  - describe_image_oci(bytes)    → Gemini 2.5 Pro vision
  - get_page_groupings_oci(...)  → Gemini 2.5 Flash page grouping
  - infer_embedding_dim()        → returns 1536
"""

import base64
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import oci
from oci.generative_ai_inference.models import (
    ChatDetails,
    EmbedTextDetails,
    GenericChatRequest,
    ImageContent,
    ImageUrl,
    OnDemandServingMode,
    SystemMessage,
    TextContent,
    UserMessage,
)

from oci_clients import (
    OCI_COMPARTMENT_ID,
    OCI_EMBED_MODEL,
    OCI_GROUPING_MODEL,
    OCI_VISION_MODEL,
    get_embed_client,
    get_grouping_client,
    get_vision_client,
)

logger = logging.getLogger("oci_pipeline")

# ---------------------------------------------------------------------------
# Embedding — Cohere Embed v4.0 (1536-dim, us-chicago-1)
# ---------------------------------------------------------------------------
EMBED_BATCH = 96  # Cohere v4 max per request


def infer_embedding_dim() -> int:
    """Probe OCI Cohere Embed v4.0 for vector dimension."""
    vecs = embed_dense_oci(["probe"])
    return len(vecs[0])


def embed_dense_oci(texts: List[str]) -> List[List[float]]:
    """
    Batch embeddings via OCI Cohere Embed v4.0.
    Returns List[List[float]] — one 1536-dim vector per input text.
    """
    client = get_embed_client()
    vectors: List[List[float]] = []

    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        for attempt in range(5):
            try:
                detail = EmbedTextDetails(
                    compartment_id=OCI_COMPARTMENT_ID,
                    serving_mode=OnDemandServingMode(model_id=OCI_EMBED_MODEL),
                    inputs=batch,
                    truncate="NONE",
                    input_type="SEARCH_DOCUMENT",
                )
                resp = client.embed_text(detail)
                vectors.extend(resp.data.embeddings)
                break
            except Exception as e:
                sleep_s = min(2**attempt, 16)
                logger.warning(f"[oci-embed] retry {attempt+1}/5: {e} (sleep {sleep_s}s)")
                time.sleep(sleep_s)
        else:
            raise RuntimeError(f"OCI embeddings failed after 5 retries for batch at offset {start}")

    return vectors


def embed_query_oci(text: str) -> List[float]:
    """Embed a single query text (uses SEARCH_QUERY input type for better retrieval)."""
    client = get_embed_client()
    detail = EmbedTextDetails(
        compartment_id=OCI_COMPARTMENT_ID,
        serving_mode=OnDemandServingMode(model_id=OCI_EMBED_MODEL),
        inputs=[text],
        truncate="NONE",
        input_type="SEARCH_QUERY",
    )
    resp = client.embed_text(detail)
    return resp.data.embeddings[0]


# ---------------------------------------------------------------------------
# Vision — Gemini 2.5 Pro (figure description)
# ---------------------------------------------------------------------------
VISION_MAX_TOKENS = 2048  # Higher than Azure's 500 — Gemini uses reasoning tokens

VISION_SYSTEM_PROMPT = """You are an expert at describing images from HR policy documents and business materials.
Provide detailed, factual descriptions of:
- Charts and graphs (include data points and trends)
- Organizational diagrams (include hierarchy and relationships)
- Process flows and workflows (describe steps)
- Tables (summarize key information)
- Forms and templates (describe structure and fields)

Focus on information that would help answer employee questions about policies and procedures.
Keep descriptions clear and concise (2-4 sentences)."""


def describe_image_oci(
    image_bytes: bytes,
    context: str = "",
    doc_name: str = "",
) -> str:
    """
    Describe an image using OCI Gemini 2.5 Pro.
    Same interface as describe_image_with_gpt4v() but uses OCI.
    """
    client = get_vision_client()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    if context:
        user_text = f"Document: {doc_name}\nCaption/Context: {context}\n\nDescribe this image:"
    else:
        user_text = f"Document: {doc_name}\n\nDescribe this image from the document:"

    chat_request = GenericChatRequest(
        messages=[
            SystemMessage(content=[TextContent(text=VISION_SYSTEM_PROMPT)]),
            UserMessage(
                content=[
                    TextContent(text=user_text),
                    ImageContent(
                        image_url=ImageUrl(url=f"data:image/png;base64,{b64_image}")
                    ),
                ]
            ),
        ],
        max_tokens=VISION_MAX_TOKENS,
        is_stream=False,
    )

    details = ChatDetails(
        compartment_id=OCI_COMPARTMENT_ID,
        serving_mode=OnDemandServingMode(model_id=OCI_VISION_MODEL),
        chat_request=chat_request,
    )

    for attempt in range(3):
        try:
            response = client.chat(details)
            choice = response.data.chat_response.choices[0]
            if choice.message and choice.message.content:
                return choice.message.content[0].text.strip()
            # Gemini may exhaust tokens on reasoning — retry with more
            logger.warning(f"[vision] empty response (finish={choice.finish_reason}), attempt {attempt+1}")
        except Exception as e:
            if attempt < 2:
                sleep_s = 2**attempt
                logger.warning(f"[vision] retry {attempt+1}/3: {e} (sleep {sleep_s}s)")
                time.sleep(sleep_s)
            else:
                logger.error(f"[vision] failed after 3 attempts: {e}")
                return f"[Image description unavailable: {str(e)[:100]}]"

    return "[Image description unavailable: empty response after retries]"


# ---------------------------------------------------------------------------
# LLM Page Grouping — Gemini 2.5 Flash
# ---------------------------------------------------------------------------
GROUPING_MAX_TOKENS = 2048


def get_page_groupings_oci(
    doc_text: str,
    pages: Dict[int, str],
    doc_name: str,
) -> Tuple[List[List[int]], bool]:
    """
    Send full document to Gemini Flash to group pages by topic.
    Same interface as get_page_groupings() from llm_semantic_chunker.py.

    Returns:
        (groups, used_fallback) — groups is e.g. [[1,2,3],[4,5],[6]]
    """
    total_pages = max(pages.keys()) if pages else 1

    if total_pages <= 1:
        return [[1]], False

    # Build page content — no truncation
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

    client = get_grouping_client()

    chat_request = GenericChatRequest(
        messages=[
            UserMessage(content=[TextContent(text=prompt)])
        ],
        temperature=0.0,
        max_tokens=GROUPING_MAX_TOKENS,
        is_stream=False,
    )

    details = ChatDetails(
        compartment_id=OCI_COMPARTMENT_ID,
        serving_mode=OnDemandServingMode(model_id=OCI_GROUPING_MODEL),
        chat_request=chat_request,
    )

    for attempt in range(3):
        try:
            response = client.chat(details)
            choice = response.data.chat_response.choices[0]
            if choice.message and choice.message.content:
                raw = choice.message.content[0].text.strip()
                groups = _parse_grouping_response(raw, total_pages)
                return groups, False
            logger.warning(f"[grouping] empty response, attempt {attempt+1}")
        except Exception as e:
            sleep_s = 2**attempt
            logger.warning(f"[grouping] retry {attempt+1}/3 for '{doc_name}': {e} (sleep {sleep_s}s)")
            time.sleep(sleep_s)

    # Fallback: one page per group
    logger.warning(f"FALLBACK: page-by-page for '{doc_name}'")
    return [[p] for p in sorted(pages.keys())], True


def _parse_grouping_response(raw: str, total_pages: int) -> List[List[int]]:
    """Parse LLM grouping response into page groups. Same logic as llm_semantic_chunker."""
    groups: List[List[int]] = []
    seen: set = set()

    group_pattern = re.compile(r"GROUP\s*:\s*([\d,\s]+)", re.IGNORECASE)
    loose_pattern = re.compile(r"^([\d,\s]+)")

    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = group_pattern.search(line)
        if not m:
            m = loose_pattern.match(line)
        if not m:
            continue

        nums = re.findall(r"\d+", m.group(1))
        group = []
        for n in nums:
            pn = int(n)
            if 1 <= pn <= total_pages and pn not in seen:
                group.append(pn)
                seen.add(pn)
        if group:
            groups.append(sorted(group))

    # Fill gaps — any missing page gets its own group
    for p in range(1, total_pages + 1):
        if p not in seen:
            groups.append([p])

    return groups
