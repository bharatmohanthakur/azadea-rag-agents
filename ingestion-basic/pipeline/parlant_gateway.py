"""
Minimal Parlant-Compatible API Gateway
Exposes a single clean endpoint for Parlant integration,
internally proxying to the main RAG server.
"""
import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Azadea Knowledge Base API",
    description="Query the Azadea HR Knowledge Base",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Main RAG server URL
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://localhost:8060")

class QueryRequest(BaseModel):
    """Request to query the knowledge base."""
    query: str

class QueryResponse(BaseModel):
    """Response from the knowledge base."""
    answer: str
    sources: str

@app.post("/query", response_model=QueryResponse, operation_id="ask_knowledge_base")
async def query_knowledge_base(request: QueryRequest):
    """
    Query the Azadea Knowledge Base.
    Use this tool to find answers about HR policies, procedures, and company guidelines.
    
    Args:
        query: The question to ask (e.g. 'What is the maternity leave policy in Lebanon?')
    
    Returns:
        answer: The response from the knowledge base
        sources: List of document sources used
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{RAG_SERVER_URL}/query",
            json={"query": request.query, "user_id": "parlant_user"}
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract and flatten sources
        sources_list = data.get("metadata", {}).get("sources", [])
        sources_str = ", ".join([
            s.get("source", "") for s in sources_list if isinstance(s, dict)
        ]) if sources_list else "No specific sources"
        
        return QueryResponse(
            answer=data.get("response", "No answer available"),
            sources=sources_str
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "parlant-gateway"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8061)
