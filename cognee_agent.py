
import os
import sys
import asyncio
import json
from typing import Literal

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langsmith import traceable

# Add PageIndex path
sys.path.insert(0, '/home/admincsp/pageindex_rag_integration')
from pageindex.utils import ChatGPT_API, ChatGPT_API_async, remove_fields, structure_to_list

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# --- Configuration ---

# Set env vars for LiteLLM/Azure keys (required for underlying calls)
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Configure Cognee Config Manually to ensure Azure usage
from cognee.infrastructure.llm.config import get_llm_config

deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

# Force update the global config
llm_config = get_llm_config()
llm_config.llm_provider = "openai"
llm_config.llm_model = f"azure/{deployment_name}"
llm_config.llm_api_key = os.getenv("AZURE_OPENAI_API_KEY")
llm_config.llm_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
llm_config.llm_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Embedding config (via env vars as Cognee might not expose it easily in LLMConfig object directly for all fields)
os.environ["EMBEDDING_PROVIDER"] = "openai"
os.environ["EMBEDDING_MODEL"] = f"azure/{embedding_deployment}"
os.environ["EMBEDDING_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["EMBEDDING_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["EMBEDDING_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Import cognee after config
try:
    import cognee
    from cognee.api.v1.search import SearchType
except ImportError:
    print("Cognee not installed. Please run: uv pip install cognee")
    sys.exit(1)


# --- PageIndex Tool ---

def create_node_mapping(tree):
    """Create a mapping from node_id to node data."""
    node_map = {}
    nodes = structure_to_list(tree)
    for node in nodes:
        if 'node_id' in node:
            node_map[node['node_id']] = node
    return node_map

@tool
async def query_pageindex_tool(question: str, document_name: str = None):
    """
    Use this tool to answer questions about specific documents by searching their hierarchical structure.
    Supports multiple documents (comma-separated).
    
    Args:
        question: The user's question about the document.
        document_name: (Optional) The specific document(s) to search. Can be a single name or comma-separated list.
    """
    if not document_name:
        # Default path for demo if no doc specified (though strict mode should prevent this)
        document_names = ["HRD - TRD - 027 - ABS Employee Incentive - G - 1_structure.json"]
    else:
        document_names = [d.strip() for d in document_name.split(',')]

    print(f"DEBUG: Processing documents: {document_names}")

    async def processed_single_doc(doc_name):
        structure_path = ""
        clean_name = doc_name.replace("Document:", "").strip()
        
        # Path resolution logic
        base_name = clean_name
        if base_name.endswith(".pdf"):
            base_name = base_name[:-4]
        elif base_name.endswith(".md"):
            base_name = base_name[:-3]
            
        # Check if it already has _structure.json (unlikely if coming from Qdrant but possible)
        if base_name.endswith("_structure.json"):
             potential_path = f"/home/admincsp/pageindex_rag_integration/results/{base_name}"
        else:
             potential_path = f"/home/admincsp/pageindex_rag_integration/results/{base_name}_structure.json"
             
        if os.path.exists(potential_path):
             structure_path = potential_path
        else:
             # Try without the chunk suffix if specific chunk file not found
             # e.g. "Doc - P - 13" -> "Doc" (heuristic)
             import re
             # Remove " - P - <digits>" or " - W - <digits>" pattern
             generic_name = re.sub(r' - [A-Z] - \d+$', '', base_name)
             if generic_name != base_name:
                 potential_path_generic = f"/home/admincsp/pageindex_rag_integration/results/{generic_name}_structure.json"
                 if os.path.exists(potential_path_generic):
                     structure_path = potential_path_generic
                     
             if not structure_path:
                 # Fallback logic for demo
                 if "incentive" in clean_name.lower():
                     structure_path = "/home/admincsp/pageindex_rag_integration/results/HRD - TRD - 027 - ABS Employee Incentive - G - 1_structure.json"
        
        if not os.path.exists(structure_path):
            return f"Error: Structure file not found for {doc_name} at {structure_path}"

        try:
            with open(structure_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            tree = doc_data.get('structure', doc_data)
            
            # Step 1: Tree Search
            tree_without_text = remove_fields(tree.copy(), fields=['text'])
            
            search_prompt = f"""
            You are given a question and a tree structure of a document.
            Find all nodes that are likely to contain the answer.
            
            Question: {question}
            
            Structure:
            {json.dumps(tree_without_text, indent=2)}
            
            Reply in JSON:
            {{
                "thinking": "reasoning",
                "node_list": ["id1", "id2"]
            }}
            """
            
            MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
            tree_search_result = await ChatGPT_API_async(MODEL, search_prompt)
            
            if "```json" in tree_search_result:
                json_str = tree_search_result.split("```json")[1].split("```")[0].strip()
            else:
                json_str = tree_search_result.strip()
            
            search_data = json.loads(json_str)
            node_list = search_data.get("node_list", [])
            
            # Step 2: Extract content
            node_map = create_node_mapping(tree)
            context_parts = []
            for node_id in node_list:
                if node_id in node_map:
                    node = node_map[node_id]
                    content = node.get('summary') or node.get('prefix_summary', '')
                    if content:
                        context_parts.append(f"[{node['title']}]\n{content}")
            
            relevant_content = "\n\n".join(context_parts)
            
            # Step 3: Generate answer for this doc
            answer_prompt = f"""
            Answer based on context from document '{doc_name}':
            Question: {question}
            Context: {relevant_content}
            """
            final_answer = await ChatGPT_API_async(MODEL, answer_prompt)
            return f"--- Answer from {doc_name} ---\n{final_answer}"

        except Exception as e:
            return f"Error processing {doc_name}: {str(e)}"

    # Run all document queries in parallel
    results = await asyncio.gather(*[processed_single_doc(name) for name in document_names])
    
    # Combine results
    combined_response = "\n\n".join(results)
    return combined_response

# --- Qdrant Tool ---

# Reuse existing logic directly from azure_doc_intelligence_qdrant
import azure_doc_intelligence_qdrant as qdrant_logic
from qdrant_client import QdrantClient

@tool
async def search_qdrant_tool(query: str):
    """
    Search for relevant documents in the Qdrant vector database.
    Returns the most relevant document filenames that can be used with the PageIndex tool.
    
    Args:
        query: The search query to find documents effectively.
    """
    try:
        # Use run_hybrid_search which initializes client internally
        # We need to ensure we can pass the collection if needed, but run_hybrid_search uses env var COLLECTION_NAME
        # We should set the env var if not set, or rely on the one set in main
        
        # Call the function directly
        hits = qdrant_logic.run_hybrid_search(query, top_k=3)
        
        results = []
        for hit in hits:
            # We need the source file name to pass to PageIndex
            source_file = hit.get("source_file")
            preview = hit.get("preview")
            results.append(f"Document: {source_file}\nSnippet: {preview}\n")
            
        return "\n".join(results)

    except Exception as e:
        print(f"DEBUG: Qdrant Search Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error searching Qdrant: {str(e)}"


# --- Cognee Tools (Sessionized) ---

# Import from the official integration package
from cognee_integration_langgraph import get_sessionized_cognee_tools

# --- Agent Setup ---

@traceable
async def main():
    # Initialize Cognee (clean slate for demo? or keep existing?)
    # await cognee.prune.prune_data() # clear old data if needed
    # await cognee.prune.prune_system(metadata=True)
    
    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )

    # Initialize Session User via langgraph-cognee
    # We define a session ID to bind the tools to a specific user context
    session_id = "interactive_session_1"
    
    # Get sessionized tools
    # This replaces the manual add_to_knowledge_tool and search_knowledge_tool
    cognee_tools = get_sessionized_cognee_tools(session_id)
    
    # Define tools (Cognee tools + specific RAG tools)
    tools = cognee_tools + [search_qdrant_tool, query_pageindex_tool]

    # Create Agent with System Prompt
    system_prompt = """You are a strict RAG agent for Azadea HR policies.
    RULES:
    1. For ANY factual question (e.g. policies, procedures, leave usage, incentives), you MUST first use `search_qdrant_tool` to find a document.
    2. If a document is found, use `query_pageindex_tool` with that document name to get the specific answer.
    3. If NO document is found by Qdrant, check your memory using the provided cognee search tool.
    4. If the information is NOT in Qdrant/PageIndex OR Cognee, you must say "I don't have this information in my documents."
    5. If the user's question is ambiguous or you find multiple relevant documents but aren't sure which one applies, ask the user for clarification before answering. (e.g. "Do you mean the UAE policy or the Lebanon policy?")
    6. DO NOT use your internal training data to answer questions about laws (like UAE labor law) or company policies. Only use the retrieved context.
    """
    
    
    # LangGraph prebuilt create_react_agent uses 'state_modifier' or 'prompt' depending on version.
    # The error suggests 'state_modifier' is unexpected.
    # Trying 'messages_modifier' or passing it as a SystemMessage in checkpointer if needed, 
    # but 'state_modifier' is the modern way. 
    # Wait, the error creates a confusion. Let's try passing it as the first message in the state.
    # Or simply: create_react_agent(llm, tools, state_modifier=...) IS correct for 0.2+, 
    # but maybe this version is older?
    # Let's try passing a SystemMessage object to 'state_modifier' or if that fails, try 'prompt'.
    
    # Actually, the error says unexpected keyword argument. 
    # Let's try the older 'messages_modifier' or just verify the version.
    # Safe bet for older versions: wrap the model or use messages_modifier.
    
    # Create Agent with Memory (Short-term conversation state)
    # This is required for multi-turn context (e.g. knowing "UAE" refers to previous topic)
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    print("Agent initialized with Memory (State+Cognee). Ready to process.")
    print(f"Session initialized for ID: {session_id}")
    
    # No more manual global user handling needed

    print("Type 'exit' or 'quit' to stop.")
    
    # Interactive Loop
    # We use a static thread_id to maintain history across input() calls in this session
    # Note: thread_id for checkpointer is separate from session_id for Cognee, but can be same concepts
    thread_id = session_id 
    config = {"configurable": {"thread_id": thread_id}}

    print("Type 'exit' or 'quit' to stop.")
    
    # Interactive Loop
    # We use a static thread_id to maintain history across input() calls in this session
    thread_id = "interactive_session_1"
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print("Agent: Thinking...")
            
            messages = [
                ("system", system_prompt),
                ("user", user_input)
            ]
            
            # Pass config with thread_id to persist state
            response = await agent_executor.ainvoke({"messages": messages}, config=config)
            
            print(f"Agent: {response['messages'][-1].content}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
