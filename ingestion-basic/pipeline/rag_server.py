#!/usr/bin/env python3
"""
RAG API Service with Graphiti Memory Integration
matches signature of api_server.py: 
POST /query {query: str} -> {response: str, metadata: dict}

Features:
- Hybrid retrieval: Qdrant (document chunks) + Graphiti (knowledge graph memory)
- Persistent memory: Saves conversations as episodes to Graphiti
- Temporal awareness: Facts include validity timestamps
- Comprehensive logging with request tracking
"""

import os
import sys
import json
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from openai import AzureOpenAI, AsyncAzureOpenAI, AsyncOpenAI

# Use existing search logic
import azure_doc_intelligence_qdrant as rag_impl

# GFM Markdown formatting
from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin

# Graphiti imports
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# ---------------------------------------------------------------------
# Logging Setup - File + Console with Rotation
# ---------------------------------------------------------------------
from logging.handlers import RotatingFileHandler

# Create logs directory
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "rag_server.log")

# Setup logger
logger = logging.getLogger("RAG-Server")
logger.setLevel(logging.INFO)

# Formatter
log_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

# File handler with rotation (10MB per file, keep 5 backups)
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"📁 Logging to file: {LOG_FILE}")

def log_request(request_id: str, step: str, data: Any, level: str = "info"):
    """Structured logging with request ID tracking."""
    msg = f"[{request_id}] {step}"
    if data:
        if isinstance(data, dict):
            msg += f" | {json.dumps(data, ensure_ascii=False, default=str)[:500]}"
        else:
            msg += f" | {str(data)[:500]}"
    
    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)

# Initialize markdown-it with GFM-like features
md = MarkdownIt("gfm-like").use(tasklists_plugin)

def format_gfm_to_html(text: str) -> str:
    """Convert markdown text to HTML using GFM-like formatting."""
    if not text or not text.strip():
        return text
    return md.render(text)

load_dotenv()

app = FastAPI(title="RAG API Service with Graphiti Memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")

# Neo4j for Graphiti
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")  # Custom database name

# Feature flags
GRAPHITI_ENABLED = os.getenv("GRAPHITI_ENABLED", "true").lower() == "true"
GRAPHITI_GROUP_ID = os.getenv("GRAPHITI_GROUP_ID", "azadea")  # Multi-tenant group ID

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# Use the multimodal collection with figure descriptions from GPT-4 Vision
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs_hybrid_azure_azadea_multimodal")

# Initialize Clients
aoai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01",
)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ---------------------------------------------------------------------
# Graphiti Memory System
# ---------------------------------------------------------------------
graphiti_instance: Optional[Graphiti] = None
graphiti_lock = asyncio.Lock()

async def get_graphiti() -> Optional[Graphiti]:
    """Get or initialize the Graphiti instance."""
    global graphiti_instance
    
    if not GRAPHITI_ENABLED:
        return None
        
    async with graphiti_lock:
        if graphiti_instance is not None:
            return graphiti_instance
            
        try:
            print("🔄 Initializing Graphiti memory system...")
            
            # Azure OpenAI clients for Graphiti
            llm_client_v1 = AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=f"{AZURE_OPENAI_ENDPOINT}openai/v1/",
            )
            
            llm_client_azure = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
            )
            
            embedding_client_azure = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
            )
            
            azure_llm_config = LLMConfig(
                model=AZURE_CHAT_DEPLOYMENT,
                small_model=AZURE_CHAT_DEPLOYMENT,
            )
            
            # Create Neo4j driver with custom database name
            neo4j_driver = Neo4jDriver(
                uri=NEO4J_URI,
                user=NEO4J_USER,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE,
            )
            logger.info(f"🔗 Connecting to Neo4j: {NEO4J_URI} (database: {NEO4J_DATABASE})")
            
            # Use custom driver with AzureOpenAILLMClient
            graphiti_instance = Graphiti(
                graph_driver=neo4j_driver,
                llm_client=AzureOpenAILLMClient(
                    azure_client=llm_client_azure,
                    config=azure_llm_config,
                    reasoning=None,  # Azure OpenAI doesn't support reasoning.effort
                    verbosity=None,
                ),
                embedder=OpenAIEmbedder(
                    config=OpenAIEmbedderConfig(
                        embedding_model=AZURE_EMBEDDING_DEPLOYMENT
                    ),
                    client=embedding_client_azure,
                ),
                cross_encoder=OpenAIRerankerClient(
                    config=LLMConfig(model=azure_llm_config.small_model),
                    client=llm_client_azure,
                ),
            )
            
            # Build indices (idempotent)
            await graphiti_instance.build_indices_and_constraints()
            logger.info(f"✅ Graphiti memory system initialized (database: {NEO4J_DATABASE})")
            return graphiti_instance
            
        except Exception as e:
            print(f"⚠️ Failed to initialize Graphiti: {e}")
            print("   Continuing without Graphiti memory...")
            return None


async def search_graphiti_memory(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Search the Graphiti knowledge graph for relevant facts using group_id for isolation."""
    graphiti = await get_graphiti()
    if not graphiti:
        return []
    
    try:
        results = await graphiti.search(
            query, 
            num_results=num_results,
            group_ids=[GRAPHITI_GROUP_ID],  # Filter by group_id for data isolation
        )
        facts = []
        for r in results:
            facts.append({
                "uuid": getattr(r, "uuid", None),
                "fact": getattr(r, "fact", ""),
                "valid_at": str(getattr(r, "valid_at", None)),
                "invalid_at": str(getattr(r, "invalid_at", None)),
                "source_node_uuid": getattr(r, "source_node_uuid", None),
                "group_id": GRAPHITI_GROUP_ID,
            })
        logger.info(f"🧠 Graphiti search with group_id={GRAPHITI_GROUP_ID} returned {len(facts)} facts")
        return facts
    except Exception as e:
        logger.error(f"⚠️ Graphiti search error: {e}")
        return []


async def save_to_graphiti_memory(user_id: str, query: str, answer: str) -> bool:
    """Save a Q&A interaction as an episode to Graphiti for long-term memory with group_id."""
    graphiti = await get_graphiti()
    if not graphiti:
        return False
    
    try:
        episode_content = f"""User ({user_id}) asked: {query}

Assistant answered: {answer}"""
        
        await graphiti.add_episode(
            name=f"conversation_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            episode_body=episode_content,
            source=EpisodeType.text,
            source_description=f"RAG conversation with user {user_id}",
            reference_time=datetime.now(timezone.utc),
            group_id=GRAPHITI_GROUP_ID,  # Assign to group_id for isolation
        )
        logger.info(f"💾 Saved conversation to Graphiti (group_id={GRAPHITI_GROUP_ID}) for user: {user_id}")
        return True
    except Exception as e:
        logger.error(f"⚠️ Failed to save to Graphiti: {e}")
        return False


# ---------------------------------------------------------------------
# Models (Matching api_server.py)
# ---------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"

class QueryResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = {}

# ---------------------------------------------------------------------
# State (In-Memory History)
# ---------------------------------------------------------------------
CONVERSATION_HISTORY: Dict[str, List[Dict[str, str]]] = {}

# Clarification state tracking across turns
# Structure: {user_id: {"turn": int, "user_responses": List[str], "rag_context": str, 
#                       "clarifying_questions": List[str], "original_query": str}}
CLARIFICATION_STATE: Dict[str, Dict[str, Any]] = {}

def get_user_history(user_id: str) -> List[Dict[str, str]]:
    if user_id not in CONVERSATION_HISTORY:
        CONVERSATION_HISTORY[user_id] = []
    return CONVERSATION_HISTORY[user_id]

class ResetRequest(BaseModel):
    user_id: str = ""

# ---------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant for Azadea HR policies and procedures.

You have access to multiple knowledge sources:
1. **Document Text**: Retrieved from HR policy documents
2. **Visual Content Descriptions**: AI-generated descriptions of charts, diagrams, workflows, and figures from documents (marked as "Visual Content" sections)
3. **Memory Facts**: Previous conversations and learned facts

When answering:
- Use information from both text and visual content descriptions
- If the answer is not in the context, politely say you don't know based on the available documents
- Keep the answer professional and concise
"""

def rewrite_query_with_history(history: List[Dict[str, str]], latest_query: str) -> str:
    """
    Rewrites the latest query based on conversation history to make it standalone.
    """
    if not history:
        return latest_query

    # Format recent history (limit to last 10 turns to save context)
    history_str = ""
    for msg in history[-10:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        history_str += f"{role}: {content}\n"

    prompt = f"""You are an AI assistant. Your task is to rewrite the latest user question into a standalone question.
    
Rules:
1. **Focus on the Immediate Context**: If the user is answering a clarifying question, combine their answer with the question.
2. **Maintain the Core Topic**: If the user asks a follow-up (e.g., "What about..."), apply it to the MAIN TOPIC discussed in previous turns (e.g., "SaaS Procurement").
3. **Resolve Pronouns**: Resolve 'it', 'they', 'that' to their referents.
4. **Do Not Hallucinate**: Only use info present in the history.

Conversation History:
{history_str}

Latest User Question: {latest_query}

Standalone Question:"""

    try:
        response = aoai_client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
            
        return rewritten
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return latest_query

@app.post("/reset")
async def reset_history(request: ResetRequest):
    if request.user_id:
        if request.user_id in CONVERSATION_HISTORY:
            CONVERSATION_HISTORY[request.user_id] = []
        if request.user_id in CLARIFICATION_STATE:
            del CLARIFICATION_STATE[request.user_id]
        return {"status": f"History and clarification state cleared for user {request.user_id}"}
    else:
        CONVERSATION_HISTORY.clear()
        CLARIFICATION_STATE.clear()
        return {"status": "History and clarification state cleared for ALL users"}

@app.get("/reset")
async def reset_history_get(user_id: Optional[str] = None):
    # Convenience for browser testing
    if user_id:
        if user_id in CONVERSATION_HISTORY:
            CONVERSATION_HISTORY[user_id] = []
        if user_id in CLARIFICATION_STATE:
            del CLARIFICATION_STATE[user_id]
        return {"status": f"History and clarification state cleared for user {user_id}"}
    else:
        CONVERSATION_HISTORY.clear()
        CLARIFICATION_STATE.clear()
        return {"status": "History cleared for ALL users"}

@app.get("/health")
async def health_check():
    """Health check endpoint with Graphiti status."""
    graphiti = await get_graphiti()
    return {
        "status": "healthy",
        "qdrant": "connected",
        "graphiti": "connected" if graphiti else "disabled",
        "graphiti_enabled": GRAPHITI_ENABLED,
    }

@app.post("/query_backup", response_model=QueryResponse)
async def query_backup_endpoint(request: QueryRequest):
    # Generate unique request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()
    
    try:
        query_text = request.query.strip()
        if not query_text:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        user_id = request.user_id or "default_user"
        history = get_user_history(user_id)
        
        # Log incoming request
        log_request(request_id, "📥 REQUEST", {
            "user_id": user_id,
            "query": query_text,
            "history_length": len(history)
        })
        
        # 0. Rewrite Query
        search_query = rewrite_query_with_history(history, query_text)
        log_request(request_id, "🔄 QUERY_REWRITE", {
            "original": query_text,
            "rewritten": search_query
        })

        # 1. Retrieve from Qdrant (document chunks)
        qdrant_start = datetime.now()
        dense_q = rag_impl.embed_dense_azure([search_query])[0]
        sparse_q = rag_impl.build_sparse_query_vector(search_query)
        
        from qdrant_client import models as qm
        
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qm.Prefetch(query=dense_q,  using=rag_impl.DENSE_NAME,  limit=50),
                qm.Prefetch(query=sparse_q, using=rag_impl.SPARSE_NAME, limit=50),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=5,
        )
        qdrant_elapsed = (datetime.now() - qdrant_start).total_seconds()
        
        sources = []
        full_context_list = []
        retrieved_images = []  # Collect images for multimodal inference
        
        for p in search_result.points:
            pl = p.payload or {}
            text_content = pl.get("text", "")
            src = pl.get("source_file", "unknown")
            page = pl.get("chunk_index", "?")
            
            full_context_list.append(f"Source: {src} (Chunk {page})\nContent: {text_content}")
            
            sources.append({
                "id": p.id,
                "score": p.score,
                "source": src,
                "text_snippet": text_content[:200],
                "has_images": pl.get("has_images", False)
            })
            
            # Extract images from payload for multimodal inference
            if pl.get("has_images") and pl.get("images"):
                for img in pl.get("images", [])[:2]:  # Limit to 2 images per chunk
                    if img.get("image_b64") and len(retrieved_images) < 3:  # Max 3 total
                        retrieved_images.append({
                            "b64": img["image_b64"],
                            "caption": img.get("caption", ""),
                            "source": src
                        })
        
        log_request(request_id, "📚 QDRANT_RESPONSE", {
            "chunks_found": len(sources),
            "images_found": len(retrieved_images),
            "elapsed_sec": round(qdrant_elapsed, 3),
            "sources": [s["source"] for s in sources]
        })
        
        # 2. Retrieve from Graphiti (knowledge graph memory)
        graphiti_start = datetime.now()
        graphiti_facts = await search_graphiti_memory(search_query, num_results=5)
        graphiti_elapsed = (datetime.now() - graphiti_start).total_seconds()
        
        log_request(request_id, "🧠 GRAPHITI_RESPONSE", {
            "facts_found": len(graphiti_facts),
            "elapsed_sec": round(graphiti_elapsed, 3),
            "facts": [f.get("fact", "")[:100] for f in graphiti_facts]
        })
        
        # Build combined context
        context_str = "\n\n".join(full_context_list)
        
        # Add Graphiti facts if available
        memory_context = ""
        if graphiti_facts:
            memory_facts_str = "\n".join(f"- {f['fact']}" for f in graphiti_facts if f.get('fact'))
            if memory_facts_str:
                memory_context = f"\n\n--- Memory Facts from Knowledge Graph ---\n{memory_facts_str}"
        
        combined_context = context_str + memory_context
        
        log_request(request_id, "🔗 COMBINED_CONTEXT", {
            "qdrant_chars": len(context_str),
            "graphiti_chars": len(memory_context),
            "total_chars": len(combined_context)
        })
        
        # 3. Generate Answer (with multimodal support)
        llm_start = datetime.now()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        
        # Inject History
        for msg in history[-5:]:
            messages.append(msg)
        
        # Build user message content (text + images for multimodal)
        if retrieved_images:
            # Multimodal message with text and images
            user_content = [
                {"type": "text", "text": f"Context:\n{combined_context}\n\nQuestion: {query_text}"}
            ]
            for img_data in retrieved_images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data['b64']}"}
                })
            messages.append({"role": "user", "content": user_content})
            log_request(request_id, "🖼️ MULTIMODAL_INFERENCE", {
                "images_included": len(retrieved_images),
                "captions": [img.get("caption", "")[:30] for img in retrieved_images]
            })
        else:
            # Text-only message
            messages.append({"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {query_text}"})
        
        completion = aoai_client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0.0,
            max_tokens=1500,
        )
        llm_elapsed = (datetime.now() - llm_start).total_seconds()
        
        answer_text = completion.choices[0].message.content
        
        log_request(request_id, "💬 LLM_RESPONSE", {
            "answer_chars": len(answer_text),
            "elapsed_sec": round(llm_elapsed, 3),
            "model": AZURE_CHAT_DEPLOYMENT,
            "multimodal": len(retrieved_images) > 0
        })
        
        # 4. Save to History (in-memory)
        history.append({"role": "user", "content": query_text})
        history.append({"role": "assistant", "content": answer_text})
        
        # 5. Save to Graphiti Memory (persistent, async - don't block response)
        asyncio.create_task(save_to_graphiti_memory(user_id, query_text, answer_text))
        
        # Format response using GFM to HTML
        formatted_response = format_gfm_to_html(answer_text)
        
        total_elapsed = (datetime.now() - start_time).total_seconds()
        log_request(request_id, "📤 RESPONSE", {
            "total_elapsed_sec": round(total_elapsed, 3),
            "response_chars": len(formatted_response),
            "qdrant_chunks": len(sources),
            "graphiti_facts": len(graphiti_facts)
        })
        
        return QueryResponse(
            response=formatted_response,
            metadata={
                "request_id": request_id,
                "sources": sources,
                "graphiti_facts_count": len(graphiti_facts),
                "memory_enabled": GRAPHITI_ENABLED,
                "elapsed_sec": round(total_elapsed, 3),
            }
        )

    except Exception as e:
        log_request(request_id, "❌ ERROR", {"error": str(e)}, level="error")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



# ---------------------------------------------------------------------
# LangGraph / Query Decomposition Integration
# ---------------------------------------------------------------------
from typing import Annotated, Literal, TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# --- Reusable RAG Search Function ---
async def run_search_for_deep_agent(query: str, user_id: str) -> Dict[str, Any]:
    """
    Executes the standard RAG search logic (Qdrant + Graphiti) and returns context + sources.
    Includes filename-based similarity search to boost documents whose names match the query.
    Returns: {"context": str, "sources": List[Dict]}
    """
    sources = []
    try:
        from qdrant_client import models as qm
        import numpy as np
        
        # 1. Embed the query
        rag_impl.embed_dense_azure([query])  # warmth
        dense_q = rag_impl.embed_dense_azure([query])[0]
        sparse_q = rag_impl.build_sparse_query_vector(query)
        
        # 2. First, get candidate documents from content search
        content_search = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qm.Prefetch(query=dense_q,  using=rag_impl.DENSE_NAME,  limit=30),
                qm.Prefetch(query=sparse_q, using=rag_impl.SPARSE_NAME, limit=30),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=15,  # Get more candidates for re-ranking
        )
        
        # 3. Extract unique source files and calculate filename similarity
        filename_scores = {}
        unique_files = set()
        for p in content_search.points:
            src_file = (p.payload or {}).get('source_file', 'unknown')
            unique_files.add(src_file)
        
        # Embed filenames and calculate similarity to query
        if unique_files:
            filenames_list = list(unique_files)
            # Use normalized filename text (remove extensions, replace separators)
            normalized_names = [f.replace('.md', '').replace('-', ' ').replace('_', ' ') for f in filenames_list]
            
            # Embed filenames
            try:
                filename_embeddings = rag_impl.embed_dense_azure(normalized_names)
                query_vec = np.array(dense_q)
                
                for i, fname in enumerate(filenames_list):
                    fname_vec = np.array(filename_embeddings[i])
                    # Cosine similarity
                    similarity = np.dot(query_vec, fname_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(fname_vec) + 1e-8)
                    filename_scores[fname] = float(similarity)
            except Exception:
                # If embedding fails, use simple keyword matching as fallback
                query_lower = query.lower()
                for fname in filenames_list:
                    fname_lower = fname.lower()
                    match_score = sum(1 for word in query_lower.split() if word in fname_lower)
                    filename_scores[fname] = match_score * 0.1  # Scale to [0, ~1]
        
        # 4. Re-rank results: combine content score with filename score
        ranked_results = []
        for p in content_search.points:
            pl = p.payload or {}
            src_file = pl.get('source_file', 'unknown')
            content_score = p.score or 0
            fname_boost = filename_scores.get(src_file, 0) * 0.3  # 30% weight for filename match
            combined_score = content_score + fname_boost
            ranked_results.append((combined_score, p))
        
        # Sort by combined score and take top 5
        ranked_results.sort(key=lambda x: x[0], reverse=True)
        top_results = ranked_results[:5]
        
        # 5. Build output
        docs_text = ""
        retrieved_images = []  # Collect images for multimodal inference
        
        for combined_score, p in top_results:
            pl = p.payload or {}
            src_file = pl.get('source_file', 'unknown')
            text_snippet = pl.get('text', '')[:600]
            docs_text += f"\n- [{src_file}]: {text_snippet}..."
            sources.append({
                "id": p.id,
                "score": round(combined_score, 4),
                "source": src_file,
                "text_snippet": text_snippet[:200],
                "filename_boost": round(filename_scores.get(src_file, 0), 4),
                "has_images": pl.get("has_images", False)
            })
            
            # Extract images from payload for multimodal inference
            if pl.get("has_images") and pl.get("images"):
                for img in pl.get("images", [])[:2]:  # Limit to 2 images per chunk
                    if img.get("image_b64") and len(retrieved_images) < 3:  # Max 3 total
                        retrieved_images.append({
                            "b64": img["image_b64"],
                            "caption": img.get("caption", ""),
                            "source": src_file
                        })
            
        # 6. Graphiti
        facts = await search_graphiti_memory(query, num_results=5)
        facts_text = "\n".join([f"- {f.get('fact')}" for f in facts])
        
        context = f"**Context for '{query}':**\n\n**Documents:**{docs_text}\n\n**Memory Facts:**\n{facts_text}"
        return {"context": context, "sources": sources, "images": retrieved_images}
        
    except Exception as e:
        return {"context": f"Error searching knowledge base for '{query}': {str(e)}", "sources": [], "images": []}


# --- LLM Client for Agent ---
agent_llm = AzureChatOpenAI(
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0
)

# --- State Definition ---
# Maximum turns for clarification before forcing a final answer
MAX_CLARIFICATION_TURNS = 3

class AgentState(TypedDict):
    original_query: str
    user_id: str
    complexity: Literal["SIMPLE", "COMPLEX", "FORMAT", "GENERIC", "DOC_PREFERENCE", "FINAL_CLARIFICATION"]
    sub_queries: List[str]
    sub_answers: List[str]
    final_answer: str
    previous_response: str  # For FORMAT path
    sources: List[Dict[str, Any]]  # Track referenced documents
    images: List[Dict[str, Any]]  # Retrieved images for multimodal inference
    # Clarification flow fields
    clarifying_questions: List[str]  # Questions to ask user for GENERIC queries
    awaiting_clarification: bool  # Flag to indicate we need user input
    user_responses: List[str]  # User's answers to clarifying questions
    rag_context_for_clarification: str  # Initial RAG context used to generate questions
    original_user_query: str  # The actual user query before doc type preference was asked
    clarification_turn: int  # Track current clarification turn (1, 2, or 3)

# --- Nodes ---

# 1. Router Node
class RouterOutput(BaseModel):
    complexity: Literal["SIMPLE", "COMPLEX", "FORMAT", "GENERIC", "DOC_PREFERENCE"] = Field(description="Classification of the query")

async def router_node(state: AgentState):
    query = state["original_query"]
    previous_response = state.get("previous_response", "")
    clarification_turn = state.get("clarification_turn", 0)
    
    # Check if we're in the middle of a clarification flow
    is_clarification_response = (
        "To help you better, I need a bit more information" in previous_response or
        "Please provide your answers" in previous_response
    )
    
    if is_clarification_response:
        # User is responding to clarifying questions
        new_turn = clarification_turn + 1
        logger.info(f"🔄 Clarification turn {new_turn} detected")
        
        if new_turn >= MAX_CLARIFICATION_TURNS:
            # At turn 3, generate final answer with all collected info
            logger.info(f"📋 Max clarification turns ({MAX_CLARIFICATION_TURNS}) reached - generating final answer")
            return {"complexity": "FINAL_CLARIFICATION", "clarification_turn": new_turn}
        else:
            # Continue clarification with updated turn count
            return {"complexity": "GENERIC", "clarification_turn": new_turn}
    
    # Check if user is responding to a document preference question
    preference_keywords = ["workflow", "policy", "guideline", "both", "1", "2", "3"]
    is_preference_response = (
        "Which type would you prefer" in previous_response and
        any(kw in query.lower() for kw in preference_keywords)
    )
    
    if is_preference_response:
        return {"complexity": "DOC_PREFERENCE"}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at routing user queries. \n"
                   "Classify the query as:\n"
                   "- 'SIMPLE' if it is specific, factual, and can be answered with a single lookup (e.g., 'What is the dress code?', 'How do I apply for leave?', 'What is the notice period?').\n"
                   "- 'COMPLEX' if it implies multiple steps, comparisons, aggregating information from different sections, or requires a comprehensive guide (e.g., 'Compare the leave policy for sick leave vs annual leave').\n"
                   "- 'FORMAT' if the user is asking to reformat, summarize differently, or change the presentation of the previous response WITHOUT needing new information (e.g., 'Put that in a table', 'Make it bullet points').\n"
                   "- 'GENERIC' if the query is ambiguous, too broad, or MISSES CRITICAL CONTEXT (like Country/Location) causing the answer to vary (e.g., 'How many days maternity leave?', 'What are the travel allowances?', 'How can I benefit from insurance?'). These need clarification."),
        ("user", "{query}")
    ])
    chain = prompt | agent_llm.with_structured_output(RouterOutput)
    result = await chain.ainvoke({"query": query})
    return {"complexity": result.complexity}

# 2. Simple Handler (Direct RAG)
class SimpleRAGOutput(BaseModel):
    answer: str = Field(description="The answer to the user query")
    status: Literal["ANSWERED", "NEEDS_CLARIFICATION"] = Field(description="Set to NEEDS_CLARIFICATION if the answer depends on missing variables (e.g. Position, Country) that the user didn't provide.")
    missing_variables: List[str] = Field(description="List of missing variables if status is NEEDS_CLARIFICATION (e.g. ['Job Position', 'Country'])")

async def simple_rag_node(state: AgentState):
    query = state["original_query"]
    user_id = state["user_id"]
    search_result = await run_search_for_deep_agent(query, user_id)
    context = search_result["context"]
    sources = search_result["sources"]
    retrieved_images = search_result.get("images", [])
    
    # Check if we have both workflow (- W) and normal documents
    workflow_sources = [s for s in sources if " - W " in s.get("source", "") or " - W-" in s.get("source", "")]
    normal_sources = [s for s in sources if s not in workflow_sources]
    
    has_workflow = len(workflow_sources) > 0
    has_normal = len(normal_sources) > 0
    
    # If we have BOTH types, ask user for preference
    if has_workflow and has_normal:
        workflow_docs = list(set([s["source"] for s in workflow_sources]))
        normal_docs = list(set([s["source"] for s in normal_sources]))
        
        response_text = (
            "I found relevant information from both **workflow documents** and **policy/guideline documents**.\n\n"
            f"**Workflow Documents** (step-by-step procedures):\n" + 
            "\n".join([f"- {doc}" for doc in workflow_docs[:3]]) + "\n\n"
            f"**Policy/Guideline Documents**:\n" + 
            "\n".join([f"- {doc}" for doc in normal_docs[:3]]) + "\n\n"
            "Which type would you prefer?\n"
            "1. **Workflow** - Detailed step-by-step process\n"
            "2. **Policy/Guideline** - General rules and information\n"
            "3. **Both** - Combined information from all sources\n\n"
            "Please reply with your preference (e.g., 'workflow', 'policy', or 'both')."
        )
        return {
            "final_answer": response_text, 
            "sources": sources,
            "images": retrieved_images,
            "awaiting_clarification": True,
            "clarifying_questions": ["Document type preference: workflow, policy, or both?"]
        }
    
    # Build messages with multimodal support if images are present
    if has_workflow and not has_normal:
        system_prompt = ("You are a helpful HR assistant. The user's query matched WORKFLOW documents which contain step-by-step procedures. "
                        "Provide a detailed, structured answer following the workflow steps. Use numbered steps where appropriate. "
                        "If images/diagrams are provided, reference them in your explanation.")
    else:
        system_prompt = ("You are a helpful HR assistant. Answer the user request based on the context provided. "
                        "If images/diagrams are provided, reference them in your explanation.\n\n"
                        "CRITICAL: Be extremely robust to malformed markdown tables. "
                        "1. HEADERS SPLIT: If a column header looks cut off (e.g., ends in '&' or starts with a lowercase letter), it belongs to the previous column. Merge them. "
                        "2. VALUES SHIFTED: If columns are split, their values might be shifted. Align them logically. "
                        "3. COMBINED HEADERS: If a header mentions multiple entities (e.g. 'Brand A & Brand B' or 'OYSHO Pull & Bear'), the values in that column apply to ALL listed entities. "
                        "4. EXTRACT VALUES: Do not complain about formatting. Use your best judgement to reconstruct the table and return the requested value.\n\n"
                        "**DYNAMIC CLARIFICATION**:\n"
                        "If the retrieved context shows that the answer varies based on specific criteria (e.g., Job Position, Country, Seniority) that the user HAS NOT provided, do **not** try to list every possible option.\n"
                        "Instead, set status to 'NEEDS_CLARIFICATION' and list the missing variables (e.g. ['Job Position']).\n"
                        "Only set this if the answer is TRULY ambiguous without that info.")
    
    # Multimodal inference if images are present
    messages = []
    messages.append(("system", system_prompt))
    
    if retrieved_images:
        # Build multimodal message with text and images
        user_content = [
            {"type": "text", "text": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        for img_data in retrieved_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_data['b64']}"}
            })
        messages.append(("user", user_content))
    else:
        messages.append(("user", f"Context:\n{context}\n\nQuestion: {query}"))
        
    chain = agent_llm.with_structured_output(SimpleRAGOutput)
    result = await chain.ainvoke(messages)
    
    if result.status == "NEEDS_CLARIFICATION":
        # Pass control to Clarifier node
        return {
            "final_answer": result.answer, # Can be empty or a transitional phrase
            "sources": sources,
            "images": retrieved_images,
            "awaiting_clarification": True,
            "rag_context_for_clarification": context, # Pass context so clarifier doesn't re-search
            "complexity": "GENERIC" # Shift complexity to GENERIC (Clarification)
        }
    else:
        return {
            "final_answer": result.answer,
            "sources": sources,
            "images": retrieved_images
        }

# 3. Decomposer (Complex Path)
class DecompositionOutput(BaseModel):
    sub_queries: List[str] = Field(description="List of 2-4 sub-questions to answer the main query.")

async def decomposer_node(state: AgentState):
    query = state["original_query"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert planner. Break down the complex query into 2-4 distinct, simpler sub-queries that, when answered, will allow you to answer the main query comprehensively. Return ONLY the list of strings."),
        ("user", "{query}")
    ])
    chain = prompt | agent_llm.with_structured_output(DecompositionOutput)
    result = await chain.ainvoke({"query": query})
    return {"sub_queries": result.sub_queries}

# 4. Executor (Complex Path)
async def executor_node(state: AgentState):
    sub_queries = state["sub_queries"]
    user_id = state["user_id"]
    answers = []
    all_sources = []
    
    # Run searches in sequence (to not overload API)
    for q in sub_queries:
        search_result = await run_search_for_deep_agent(q, user_id)
        context_str = search_result["context"]
        all_sources.extend(search_result["sources"])
        answers.append(f"### Q: {q}\n{context_str}")
        
    return {"sub_answers": answers, "sources": all_sources}

# 5. Synthesizer (Complex Path)
async def synthesizer_node(state: AgentState):
    original_query = state["original_query"]
    sub_answers = state["sub_answers"]
    
    combined_context = "\n\n".join(sub_answers)
    
    messages = [
        ("system", "You are a helpful HR expert. You have gathered information for a complex user request. "
                   "Synthesize the provided sub-answers into a cohesive final report.\n\n"
                   "**CRITICAL INSTRUCTION**:\n"
                   "1. **Direct Answer First**: Start by directly answering the user's ORIGINAL request using the synthesized information.\n"
                   "2. **Supporting Details**: Then, provide the detailed breakdown based on the sub-queries investigating specific aspects.\n"
                   "3. Do not explicitly mention 'sub-queries' or 'step 1', just weave the information together naturally."),
        ("user", f"Original Request: {original_query}\n\nGathered Information:\n{combined_context}")
    ]
    response = await agent_llm.ainvoke(messages)
    return {"final_answer": response.content}

# 6. Format Handler (FORMAT Path - No RAG, just reformat previous response)
async def format_handler_node(state: AgentState):
    query = state["original_query"]
    previous_response = state.get("previous_response", "")
    
    if not previous_response:
        return {"final_answer": "I don't have a previous response to reformat. Please ask a question first."}
    
    messages = [
        ("system", "You are a helpful assistant. The user wants you to reformat or re-present a previous response. "
                   "Apply the requested formatting changes to the content provided. Keep the same information, just change how it's presented."),
        ("user", f"Previous Response:\n{previous_response}\n\nUser Request: {query}")
    ]
    response = await agent_llm.ainvoke(messages)
    return {"final_answer": response.content}

# 7. Clarifier Node (GENERIC Path - Ask clarifying questions based on RAG data)
class ClarificationOutput(BaseModel):
    questions: List[str] = Field(description="List of 2-4 clarifying questions to ask the user")
    categories_found: List[str] = Field(description="Categories/options found in the knowledge base")

async def clarifier_node(state: AgentState):
    """
    For GENERIC queries: Fetch initial RAG data, analyze what options/categories exist,
    and generate targeted clarifying questions based on available data.
    Shows turn count to user (e.g., "Turn 1 of 3").
    """
    query = state["original_query"]
    user_id = state["user_id"]
    clarification_turn = state.get("clarification_turn", 1)
    user_responses = state.get("user_responses", [])
    
    # Add current response to user_responses if this is a follow-up turn
    if clarification_turn > 1:
        user_responses.append(query)
    
    # Check if we already have context (passed from simple_rag_node fallback)
    context = state.get("rag_context_for_clarification")
    sources = state.get("sources", [])
    
    # If not, Fetch initial RAG data (standard GENERIC path)
    if not context:
        search_result = await run_search_for_deep_agent(query, user_id)
        context = search_result["context"]
        sources = search_result["sources"]
    
    # Generate clarifying questions based on what's in the data and previous responses
    previous_responses_text = ""
    if user_responses:
        previous_responses_text = f"\n\nUser's previous responses: {'; '.join(user_responses)}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an HR assistant helping to clarify a user's generic question.

Based on the retrieved context from our knowledge base, generate 2-4 targeted clarifying questions.

IMPORTANT RULES:
1. Questions should be based on ACTUAL OPTIONS/CATEGORIES found in the context
2. Questions should help narrow down exactly what the user needs
3. Format questions as a numbered list
4. Be specific - use real category names from the context (e.g., "health insurance", "life insurance", "dental")
5. Keep questions concise and clear
6. If the user has already provided some answers, ask follow-up questions that haven't been answered yet

Example: If user asks "How can I benefit from insurance?" and context mentions health, life, and dental insurance:
- What type of insurance are you interested in: health insurance, life insurance, or dental insurance?
- Are you asking about coverage limits, enrollment process, or claim procedures?"""),
        ("user", f"User's generic question: {query}{previous_responses_text}\\n\\nAvailable context from knowledge base:\\n{context}\\n\\nGenerate clarifying questions:")
    ])
    
    chain = prompt | agent_llm.with_structured_output(ClarificationOutput)
    result = await chain.ainvoke({"query": query, "context": context})
    
    # Format the clarifying questions as the response with turn indicator
    questions_text = "\\n".join([f"{i+1}. {q}" for i, q in enumerate(result.questions)])
    turns_remaining = MAX_CLARIFICATION_TURNS - clarification_turn
    turn_indicator = f"**(Turn {clarification_turn} of {MAX_CLARIFICATION_TURNS})**"
    
    if turns_remaining == 1:
        notice = "\\n\\n*Note: This is my last clarifying question. On your next response, I'll provide the best answer I can with the information gathered.*"
    else:
        notice = ""
    
    response_text = f"{turn_indicator}\\n\\nTo help you better, I need a bit more information:\\n\\n{questions_text}\\n\\nPlease provide your answers and I'll give you a detailed response.{notice}"
    
    return {
        "final_answer": response_text,
        "clarifying_questions": result.questions,
        "awaiting_clarification": True,
        "rag_context_for_clarification": context,
        "sources": sources,
        "user_responses": user_responses,
        "clarification_turn": clarification_turn
    }

# 7b. Final Clarification Node (Turn 3 - Collate all responses and generate final answer)
async def final_clarification_node(state: AgentState):
    """
    At turn 3 (MAX_CLARIFICATION_TURNS), collate all user clarification responses,
    generate a final refined question, and get the answer from RAG.
    """
    original_query = state["original_query"]
    user_id = state["user_id"]
    user_responses = state.get("user_responses", [])
    context = state.get("rag_context_for_clarification", "")
    clarifying_questions = state.get("clarifying_questions", [])
    
    # Add the final response to user_responses
    user_responses.append(original_query)
    
    logger.info(f"📋 Final clarification - Collating responses: {user_responses}")
    
    # Step 1: Generate a final refined question from all collected information
    qa_pairs = ""
    for i, response in enumerate(user_responses):
        if i < len(clarifying_questions):
            qa_pairs += f"Q: {clarifying_questions[i]}\\nA: {response}\\n\\n"
        else:
            qa_pairs += f"User additional info: {response}\\n\\n"
    
    refine_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an HR assistant. Based on the clarification dialog below, generate a single, 
detailed, standalone question that incorporates ALL the user's clarifications and preferences.

The question should be specific enough to get a precise answer from our HR knowledge base.

Example:
- Original: "What is the leave policy?"
- After clarifications about country (Lebanon) and type (maternity)
- Final question: "What is the maternity leave policy for employees in Lebanon?"

Output ONLY the refined question, nothing else."""),
        ("user", f"Original question from user: {state.get('original_user_query', original_query)}\\n\\nClarification dialog:\\n{qa_pairs}\\n\\nGenerate the final refined question:")
    ])
    
    refine_response = await agent_llm.ainvoke(refine_prompt.format_messages())
    final_question = refine_response.content.strip()
    
    logger.info(f"📝 Final refined question: {final_question}")
    
    # Step 2: Search with the refined question
    search_result = await run_search_for_deep_agent(final_question, user_id)
    new_context = search_result["context"]
    sources = search_result["sources"]
    retrieved_images = search_result.get("images", [])
    
    # Step 3: Generate the final answer
    messages = [
        ("system", """You are a helpful HR assistant. The user has gone through a clarification process 
and we now have a specific question. Provide a comprehensive, accurate answer based on the context.

If images/diagrams are provided, reference them in your explanation.
If the exact answer is not in the context, provide the closest relevant information and note any limitations."""),
    ]
    
    if retrieved_images:
        user_content = [
            {"type": "text", "text": f"Context:\\n{new_context}\\n\\nUser's clarified question: {final_question}"}
        ]
        for img_data in retrieved_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_data['b64']}"}
            })
        messages.append(("user", user_content))
    else:
        messages.append(("user", f"Context:\\n{new_context}\\n\\nUser's clarified question: {final_question}"))
    
    response = await agent_llm.ainvoke(messages)
    
    return {
        "final_answer": response.content,
        "sources": sources,
        "images": retrieved_images,
        "awaiting_clarification": False,
        "clarification_turn": MAX_CLARIFICATION_TURNS
    }

# 8. Answer Relevance Layer - Aligns response to user intent
class AnswerRelevanceOutput(BaseModel):
    is_relevant: bool = Field(description="True if the answer already addresses the user's question/intent, False if it needs refinement")
    refined_answer: str = Field(description="The refined answer if is_relevant is False, otherwise the original answer unchanged")
    relevance_reason: str = Field(description="Brief explanation of the relevance assessment")

# Common greetings and casual messages that should get friendly responses, not clarifying questions
GREETING_PATTERNS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
    "howdy", "greetings", "what's up", "sup", "yo", "hiya"
]
CASUAL_PATTERNS = [
    "thanks", "thank you", "ok", "okay", "bye", "goodbye", "see you", 
    "great", "cool", "nice", "awesome", "perfect", "got it"
]
EMOTIONAL_PATTERNS = [
    "lonely", "sad", "depressed", "stressed", "anxious", "worried", "upset",
    "happy", "excited", "confused", "frustrated", "tired", "bored"
]

def is_greeting_or_casual(query: str) -> bool:
    """Check if the query is a greeting, casual message, or emotional expression."""
    query_lower = query.lower().strip()
    
    # Check greetings
    for pattern in GREETING_PATTERNS:
        if query_lower == pattern or query_lower.startswith(pattern + " ") or query_lower.startswith(pattern + ","):
            return True
    
    # Check casual messages
    for pattern in CASUAL_PATTERNS:
        if pattern in query_lower:
            return True
    
    # Check emotional expressions
    for pattern in EMOTIONAL_PATTERNS:
        if pattern in query_lower:
            return True
    
    # Very short messages without HR keywords are likely casual
    if len(query_lower.split()) <= 3:
        hr_keywords = ["leave", "policy", "salary", "bonus", "insurance", "benefits", "vacation", 
                       "maternity", "paternity", "sick", "annual", "hr", "employee", "work", "job"]
        if not any(kw in query_lower for kw in hr_keywords):
            return True
    
    return False

async def answer_relevance_node(state: AgentState):
    """
    Answer Relevance Layer: Evaluates if the final response aligns with user intent.
    - For greetings/casual messages -> refine to friendly response (skip clarifying questions)
    - For actual HR queries needing clarification -> preserve clarifying questions
    - For complete answers -> validate relevance
    """
    original_query = state.get("original_query", "")
    final_answer = state.get("final_answer", "")
    awaiting_clarification = state.get("awaiting_clarification", False)
    
    # Skip if no answer to evaluate
    if not final_answer:
        return state
    
    # Skip for very short answers (likely error messages or simple confirmations)
    if len(final_answer) < 50:
        return state
    
    # IMPORTANT: Only refine if the query is a greeting/casual message
    # For actual HR queries that need clarification, preserve the clarifying questions
    if not is_greeting_or_casual(original_query):
        # This is an actual HR query - don't interfere with clarification
        if awaiting_clarification:
            logger.info(f"✅ Answer relevance: Preserving clarification for HR query")
            return state
        
        # For non-clarification HR answers, do a quick relevance check
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Answer Relevance Evaluator for an HR knowledge base.

Check if the answer properly addresses the HR question.

RULES:
- If the answer addresses the user's HR question well: is_relevant=True, return answer UNCHANGED
- If the answer is completely off-topic: is_relevant=False, provide a refined answer
- Do NOT change answers that are already relevant
- Preserve all factual HR information"""),
            ("user", f"""User Question: {original_query}

Answer: {final_answer}

Is this answer relevant to the question?""")
        ])
    else:
        # This is a greeting/casual message - refine to appropriate response
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Answer Relevance Evaluator.

The user sent a greeting or casual message. Check if the response is appropriate.

RULES:
- GREETINGS (hi, hello, hey): Should get a friendly greeting back, NOT clarifying questions
- CASUAL (thanks, ok, bye): Should get natural conversational response
- EMOTIONAL (feeling lonely, stressed): Should get empathetic response

If the response is clarifying questions for a simple greeting, that is NOT relevant - fix it."""),
            ("user", f"""User Message: {original_query}

Response: {final_answer}

If this gave clarifying questions for a simple greeting, fix it with an appropriate response.""")
        ])
    
    try:
        chain = prompt | agent_llm.with_structured_output(AnswerRelevanceOutput)
        result = await chain.ainvoke({"query": original_query, "answer": final_answer})
        
        if result.is_relevant:
            logger.info(f"✅ Answer relevance check: ALIGNED - {result.relevance_reason[:100]}")
            return state
        else:
            logger.info(f"🔄 Answer relevance check: REFINED - {result.relevance_reason[:100]}")
            return {"final_answer": result.refined_answer, "awaiting_clarification": False}
            
    except Exception as e:
        logger.warning(f"⚠️ Answer relevance check failed, using original: {e}")
        return state


# 9. Document Preference Handler (DOC_PREFERENCE Path)
async def doc_preference_handler_node(state: AgentState):
    """
    Handle user's response to document type preference question.
    Extract original query from history and answer based on preferred doc type.
    """
    preference = state["original_query"].lower()
    user_id = state["user_id"]
    original_user_query = state.get("original_user_query", "")
    
    # Determine which doc types to use
    use_workflow = "workflow" in preference or "1" in preference
    use_policy = "policy" in preference or "guideline" in preference or "2" in preference
    use_both = "both" in preference or "3" in preference
    
    # Re-search with the original query
    search_result = await run_search_for_deep_agent(original_user_query, user_id)
    context = search_result["context"]
    sources = search_result["sources"]
    
    # Filter sources based on preference
    if use_workflow and not use_both:
        filtered_sources = [s for s in sources if " - W " in s.get("source", "") or " - W-" in s.get("source", "")]
        doc_type_instruction = "Focus on WORKFLOW documents which contain step-by-step procedures. Provide detailed steps."
    elif use_policy and not use_both:
        filtered_sources = [s for s in sources if " - W " not in s.get("source", "") and " - W-" not in s.get("source", "")]
        doc_type_instruction = "Focus on POLICY/GUIDELINE documents. Provide general rules and information."
    else:  # both
        filtered_sources = sources
        doc_type_instruction = "Use ALL available documents. Provide comprehensive information including both procedures and policies."
    
    # Generate answer
    messages = [
        ("system", f"You are a helpful HR assistant. {doc_type_instruction}"),
        ("user", f"Context:\n{context}\n\nQuestion: {original_user_query}")
    ]
    response = await agent_llm.ainvoke(messages)
    return {"final_answer": response.content, "sources": filtered_sources}

# --- Graph Contruction ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("simple_rag", simple_rag_node)
workflow.add_node("decomposer", decomposer_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("format_handler", format_handler_node)
workflow.add_node("clarifier", clarifier_node)
workflow.add_node("final_clarification", final_clarification_node)  # Turn 3 - final answer
workflow.add_node("doc_preference_handler", doc_preference_handler_node)
workflow.add_node("answer_relevance", answer_relevance_node)  # Answer relevance layer

workflow.add_edge(START, "router")

def route_logic(state: AgentState):
    if state["complexity"] == "COMPLEX":
        return "decomposer"
    elif state["complexity"] == "FORMAT":
        return "format_handler"
    elif state["complexity"] == "GENERIC":
        return "clarifier"
    elif state["complexity"] == "DOC_PREFERENCE":
        return "doc_preference_handler"
    elif state["complexity"] == "FINAL_CLARIFICATION":
        return "final_clarification"
    return "simple_rag"

workflow.add_conditional_edges("router", route_logic)

workflow.add_edge("decomposer", "executor")
workflow.add_edge("executor", "synthesizer")
# Route synthesizer through answer relevance layer
workflow.add_edge("synthesizer", "answer_relevance")

# Route format_handler through answer relevance layer
workflow.add_edge("format_handler", "answer_relevance")

# Route clarifier through answer relevance layer (to catch greetings/casual messages)
workflow.add_edge("clarifier", "answer_relevance")

# Route doc_preference_handler through answer relevance layer
workflow.add_edge("doc_preference_handler", "answer_relevance")

# Route final_clarification through answer relevance layer
workflow.add_edge("final_clarification", "answer_relevance")

# Answer relevance layer goes to END
workflow.add_edge("answer_relevance", END)

def check_simple_rag_status(state: AgentState):
    """Check if simple_rag decided it needs clarification."""
    if state.get("complexity") == "GENERIC" and state.get("awaiting_clarification"):
         return "clarifier"
    # Route through answer relevance layer if not awaiting clarification
    return "answer_relevance"

workflow.add_conditional_edges("simple_rag", check_simple_rag_status)

deep_agent_app = workflow.compile()


@app.post("/query", response_model=QueryResponse, operation_id="query_knowledge_base")
async def query_endpoint(request: QueryRequest):
    """
    Queries the Azadea Knowledge Base.
    Use this tool to fetch answers for specific employee questions.
    Inputs:
    - query: The standalone question (e.g. 'What is the maternity leave policy in Lebanon?')
    - user_id: (Optional) The user's ID.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()
    
    try:
        query_text = request.query.strip()
        user_id = request.user_id or "default_user"
        
        log_request(request_id, "🤖 DEEP_AGENT_START", {"query": query_text})

        # Fetch history and rewrite query for context
        history = get_user_history(user_id)
        rewritten_query = rewrite_query_with_history(history, query_text)
        
        if rewritten_query != query_text:
            log_request(request_id, "🔄 DEEP_QUERY_REWRITE", {
                "original": query_text,
                "rewritten": rewritten_query
            })

        # Extract previous assistant response for FORMAT path
        previous_response = ""
        original_user_query = ""
        if history:
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    previous_response = msg.get("content", "")
                    break
            # Extract original user query (the one before the preference question was asked)
            # This is the second-to-last user message if the last assistant message was a preference question
            if "Which type would you prefer" in previous_response:
                user_messages = [m for m in history if m.get("role") == "user"]
                if len(user_messages) >= 1:
                    original_user_query = user_messages[-1].get("content", "")

        # Get clarification state for this user (persisted across turns)
        clarification_state = CLARIFICATION_STATE.get(user_id, {})
        clarification_turn = clarification_state.get("turn", 0)
        stored_user_responses = clarification_state.get("user_responses", [])
        stored_context = clarification_state.get("rag_context", "")
        stored_questions = clarification_state.get("clarifying_questions", [])
        stored_original_query = clarification_state.get("original_query", "")
        
        # Log clarification state
        log_request(request_id, "📊 CLARIFICATION_STATE", {
            "turn": clarification_turn,
            "user_responses": len(stored_user_responses),
            "has_context": bool(stored_context)
        })

        # Initial state used rewritten query for better routing and retrieval
        initial_state = {
            "original_query": rewritten_query,
            "user_id": user_id,
            "complexity": "SIMPLE",
            "sub_queries": [],
            "sub_answers": [],
            "final_answer": "",
            "previous_response": previous_response,
            "sources": [],
            "images": [],  # Multimodal images
            # Clarification flow fields
            "clarifying_questions": stored_questions,
            "awaiting_clarification": False,
            "user_responses": stored_user_responses,
            "rag_context_for_clarification": stored_context,
            "original_user_query": stored_original_query if stored_original_query else original_user_query,
            "clarification_turn": clarification_turn
        }
        
        # Invoke LangGraph
        result = await deep_agent_app.ainvoke(initial_state)
        answer_text = result.get("final_answer", "No answer generated.")
        complexity = result.get("complexity", "UNKNOWN")
        awaiting_clarification = result.get("awaiting_clarification", False)
        
        # Persist clarification state for next turn
        if awaiting_clarification:
            new_turn = result.get("clarification_turn", 1)
            CLARIFICATION_STATE[user_id] = {
                "turn": new_turn,
                "user_responses": result.get("user_responses", []),
                "rag_context": result.get("rag_context_for_clarification", ""),
                "clarifying_questions": result.get("clarifying_questions", []),
                "original_query": result.get("original_user_query", "") or query_text
            }
            log_request(request_id, "💾 CLARIFICATION_STATE_SAVED", {
                "turn": new_turn,
                "user_responses": len(result.get("user_responses", []))
            })
        else:
            # Clear clarification state when done (got final answer)
            if user_id in CLARIFICATION_STATE:
                del CLARIFICATION_STATE[user_id]
                log_request(request_id, "🗑️ CLARIFICATION_STATE_CLEARED", {"user_id": user_id})
        
        # Log & Save History
        total_elapsed = (datetime.now() - start_time).total_seconds()
        
        log_request(request_id, "🤖 DEEP_AGENT_END", {
            "elapsed_sec": round(total_elapsed, 3),
            "complexity": complexity,
            "sub_queries": len(result.get("sub_queries", [])),
            "response_length": len(answer_text),
            "awaiting_clarification": awaiting_clarification
        })

        # Update server-side history (Critical for context rewriting)
        history.append({"role": "user", "content": query_text})
        history.append({"role": "assistant", "content": answer_text})

        # Async save to graphiti
        asyncio.create_task(save_to_graphiti_memory(user_id, query_text, answer_text))
        
        return QueryResponse(
            response=format_gfm_to_html(answer_text),
            metadata={
                "request_id": request_id,
                "agent": "LangGraph Decomposition",
                "complexity": complexity,
                "sub_queries": result.get("sub_queries", []),
                "sources": result.get("sources", []),
                "elapsed_sec": round(total_elapsed, 3)
            }
        )
        
    except Exception as e:
        log_request(request_id, "❌ DEEP_AGENT_ERROR", {"error": str(e)}, level="error")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# Parlant-Compatible Endpoint (Simple Schema)
# ---------------------------------------------------------------------
class ParlantQueryRequest(BaseModel):
    """Minimal request schema for Parlant compatibility - no Optional types."""
    query: str

class ParlantQueryResponse(BaseModel):
    """Minimal response schema for Parlant compatibility."""
    answer: str
    sources: str

@app.post("/parlant_query", response_model=ParlantQueryResponse, operation_id="ask_knowledge_base")
async def parlant_query_endpoint(request: ParlantQueryRequest):
    """
    Query the Azadea Knowledge Base.
    Use this tool to find answers about HR policies, procedures, and company guidelines.
    
    Args:
        query: The question to ask (e.g. 'What is the maternity leave policy in Lebanon?')
    
    Returns:
        answer: The response from the knowledge base
        sources: List of document sources used
    """
    # Call the main query logic internally
    internal_request = QueryRequest(query=request.query, user_id="parlant_user")
    result = await query_endpoint(internal_request)
    
    # Flatten sources to a simple string for Parlant
    sources_list = result.metadata.get("sources", [])
    sources_str = ", ".join([s.get("source", "") for s in sources_list if isinstance(s, dict)]) if sources_list else "No specific sources"
    
    return ParlantQueryResponse(
        answer=result.response,
        sources=sources_str
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup Graphiti connection on shutdown."""
    global graphiti_instance
    if graphiti_instance:
        try:
            await graphiti_instance.close()
            print("✅ Graphiti connection closed")
        except Exception:
            pass
        graphiti_instance = None

if __name__ == "__main__":
    import uvicorn
    # Using port 8060 to avoid conflicts
    uvicorn.run(app, host="0.0.0.0", port=8060)
