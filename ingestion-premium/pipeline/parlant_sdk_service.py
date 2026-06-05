"""
Parlant Native SDK Service
Uses Parlant's native Python SDK to create tools that wrap the RAG query functionality.
This avoids OpenAPI schema compatibility issues.
"""
import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

# Import Parlant SDK 
import parlant.sdk as p


# RAG Server configuration
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://localhost:8060")


@p.tool
async def query_knowledge_base(context: p.ToolContext, query: str) -> p.ToolResult:
    """
    Query the Azadea Knowledge Base for HR policies and procedures.
    
    Args:
        context: Parlant tool context
        query: The question to ask (e.g. 'What is the maternity leave policy in Lebanon?')
    
    Returns:
        ToolResult with the answer from the knowledge base
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{RAG_SERVER_URL}/query",
                json={"query": query, "user_id": "parlant_user"}
            )
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("response", "No answer available")
            sources_list = data.get("metadata", {}).get("sources", [])
            sources_str = ", ".join([
                s.get("source", "") for s in sources_list if isinstance(s, dict)
            ]) if sources_list else "No specific sources"
            
            return p.ToolResult(f"{answer}\n\nSources: {sources_str}")
            
        except Exception as e:
            return p.ToolResult(f"Error querying knowledge base: {str(e)}")


@p.tool  
async def get_current_datetime(context: p.ToolContext) -> p.ToolResult:
    """Get the current date and time."""
    from datetime import datetime
    return p.ToolResult(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


async def main():
    """Start the Parlant server with our HR agent."""
    
    # Load guidelines from clustered extraction
    import json
    guidelines_file = "clustered_guidelines.json"
    
    # Set up Azure OpenAI environment variables that Parlant expects
    # Parlant expects AZURE_API_KEY and AZURE_ENDPOINT for Azure mode
    os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")
    os.environ["AZURE_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    os.environ["AZURE_DEPLOYMENT"] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    
    # Use Azure NLP service via the enum, on different ports to avoid conflicts
    async with p.Server(nlp_service=p.NLPServices.azure, port=8801, tool_service_port=8821) as server:
        # Create the HR agent
        agent = await server.create_agent(
            name="AzadeaHR",
            description="Helpful assistant for Azadea HR policies and procedures"
        )
        
        # Add a context variable for current time
        await agent.create_variable(
            name="current-datetime",
            tool=get_current_datetime
        )
        
        # Core guidelines
        await agent.create_guideline(
            condition="User asks about HR policies, leave, insurance, or company procedures",
            action="Query the knowledge base to find the answer. If the user hasn't specified their country or role, ask for clarification first.",
            tools=[query_knowledge_base]
        )
        
        await agent.create_guideline(
            condition="User asks a broad question without specifying country or role context",
            action="Ask clarifying questions to determine the user's country (e.g., 'Are you in UAE, Lebanon, or another country?') and role before querying the knowledge base."
        )
        
        await agent.create_guideline(
            condition="User provides specific context (country, role) with their question",
            action="Query the knowledge base with the full context and provide a clear, helpful response.",
            tools=[query_knowledge_base]
        )
        
        # Load additional guidelines from clustered extraction
        if os.path.exists(guidelines_file):
            print(f"Loading guidelines from {guidelines_file}...")
            with open(guidelines_file, "r") as f:
                clustered_guidelines = json.load(f)
                
            # Add top 15 most relevant clustered guidelines
            for g in clustered_guidelines[:15]:
                await agent.create_guideline(
                    condition=g["condition"],
                    action=g["action"],
                    tools=[query_knowledge_base] if "ask" not in g["action"].lower()[:20] else []
                )
                print(f"  Added guideline for cluster: {g.get('cluster', 'unknown')}")
        
        print("\n✅ Parlant server started!")
        print("   Agent: AzadeaHR")
        print("   Test playground: http://localhost:8800")
        print("   Press Ctrl+C to stop")
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
