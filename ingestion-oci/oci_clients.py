"""
OCI GenAI client factory — shared by all OCI pipeline functions.

Provides:
  - get_embed_client()   → Cohere Embed v4.0 (us-chicago-1)
  - get_chat_client()    → Gemini Flash/Pro (eu-frankfurt-1)
  - OCI_COMPARTMENT_ID   → BrainShift compartment
"""

import os
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient

# ---------------------------------------------------------------------------
# OCI Auth Config
# ---------------------------------------------------------------------------
# OCI identity comes ONLY from the environment (.env / secret manager) — never
# hardcoded in source. Set OCI_USER, OCI_FINGERPRINT, OCI_TENANCY,
# OCI_COMPARTMENT_ID, OCI_KEY_FILE (and optionally OCI_REGION) in the environment.
_OCI_CONFIG = {
    "user": os.getenv("OCI_USER"),
    "fingerprint": os.getenv("OCI_FINGERPRINT"),
    "tenancy": os.getenv("OCI_TENANCY"),
    "region": os.getenv("OCI_REGION", "eu-frankfurt-1"),
    "key_file": os.getenv("OCI_KEY_FILE"),
}

OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")

if not all([_OCI_CONFIG["user"], _OCI_CONFIG["fingerprint"], _OCI_CONFIG["tenancy"],
            _OCI_CONFIG["key_file"], OCI_COMPARTMENT_ID]):
    raise RuntimeError(
        "Missing OCI identity. Set OCI_USER, OCI_FINGERPRINT, OCI_TENANCY, "
        "OCI_KEY_FILE and OCI_COMPARTMENT_ID in the environment (.env)."
    )

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
OCI_EMBED_MODEL = os.getenv("OCI_EMBED_MODEL", "cohere.embed-v4.0")
OCI_EMBED_REGION = os.getenv("OCI_EMBED_REGION", "us-chicago-1")
OCI_VISION_MODEL = os.getenv("OCI_VISION_MODEL", "google.gemini-2.5-pro")
OCI_VISION_REGION = os.getenv("OCI_VISION_REGION", "eu-frankfurt-1")
OCI_GROUPING_MODEL = os.getenv("OCI_GROUPING_MODEL", "google.gemini-2.5-pro")
OCI_GROUPING_REGION = os.getenv("OCI_GROUPING_REGION", "eu-frankfurt-1")

# ---------------------------------------------------------------------------
# Client factories (lazy singletons)
# ---------------------------------------------------------------------------
_embed_client = None
_chat_client_fra = None
_chat_client_chi = None


def _make_client(region: str) -> GenerativeAiInferenceClient:
    cfg = dict(_OCI_CONFIG)
    cfg["region"] = region
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    return GenerativeAiInferenceClient(
        config=cfg,
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 300),
    )


def get_embed_client() -> GenerativeAiInferenceClient:
    """Client for Cohere Embed v4.0 — us-chicago-1."""
    global _embed_client
    if _embed_client is None:
        _embed_client = _make_client(OCI_EMBED_REGION)
    return _embed_client


def get_vision_client() -> GenerativeAiInferenceClient:
    """Client for Gemini 2.5 Pro vision — eu-frankfurt-1."""
    global _chat_client_fra
    if _chat_client_fra is None:
        _chat_client_fra = _make_client(OCI_VISION_REGION)
    return _chat_client_fra


def get_grouping_client() -> GenerativeAiInferenceClient:
    """Client for Gemini 2.5 Pro page grouping — eu-frankfurt-1."""
    if OCI_GROUPING_REGION == OCI_VISION_REGION:
        return get_vision_client()
    global _chat_client_chi
    if _chat_client_chi is None:
        _chat_client_chi = _make_client(OCI_GROUPING_REGION)
    return _chat_client_chi


# ---------------------------------------------------------------------------
# OpenAI-compatible clients (for RAG server: classification, rewriting, answers)
# Uses oci-openai package for full OpenAI SDK compatibility
# ---------------------------------------------------------------------------
OCI_CONFIG_FILE = os.getenv("OCI_CONFIG_FILE", "/home/admincsp/graphiti_fixed_test/ingestion-oci/oci_config")
OCI_CHAT_MODEL = os.getenv("OCI_CHAT_MODEL", "google.gemini-2.5-flash")

_oci_openai_client = None
_agent_llm = None


def get_oci_openai_client():
    """OpenAI-compatible client backed by OCI GenAI. For UnifiedClassifier + rewrite."""
    global _oci_openai_client
    if _oci_openai_client is None:
        from oci_openai import OciOpenAI, OciUserPrincipalAuth
        auth = OciUserPrincipalAuth(config_file=OCI_CONFIG_FILE)
        _oci_openai_client = OciOpenAI(
            auth=auth,
            region="eu-frankfurt-1",
            compartment_id=OCI_COMPARTMENT_ID,
        )
    return _oci_openai_client


def get_agent_llm():
    """LangChain ChatOpenAI backed by OCI GenAI. For answer generation (ainvoke + astream)."""
    global _agent_llm
    if _agent_llm is None:
        import httpx
        from oci_openai import OciUserPrincipalAuth
        from langchain_openai import ChatOpenAI

        auth = OciUserPrincipalAuth(config_file=OCI_CONFIG_FILE)
        base_url = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/openai/v1/"

        http_client = httpx.Client(
            auth=auth,
            headers={"CompartmentId": OCI_COMPARTMENT_ID},
        )
        async_http = httpx.AsyncClient(
            auth=auth,
            headers={"CompartmentId": OCI_COMPARTMENT_ID},
        )

        _agent_llm = ChatOpenAI(
            model=OCI_CHAT_MODEL,
            api_key="OCI",
            base_url=base_url,
            http_client=http_client,
            http_async_client=async_http,
            temperature=0,
            max_tokens=2048,
        )
    return _agent_llm
