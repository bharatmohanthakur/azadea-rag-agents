
import subprocess
import time
import sys
import os
import json

def run_parlant_cmd(args):
    """Runs a parlant CLI command."""
    cmd = ["parlant", "-s", "http://localhost:8800"] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout.strip()}")
    return result

def setup_parlant_poc():
    print("--- Setting up Parlant POC Agent ---")
    
    # 1. Create Agent
    print("\n1. Creating Agent 'AzadeaHR'...")
    # Checking if exists first would be good, but 'create' might fail or warn.
    # We'll just try to create.
    run_parlant_cmd(["agent", "create", "--name", "AzadeaHR"])

    # 2. Add Service (RAG API)
    print("\n2. Registering RAG Service...")
    # Delete existing if any (to refresh schema)
    run_parlant_cmd(["service", "delete", "--name", "AzadeaRAG"])
    
    # Assuming parlant_gateway is running on 8061
    run_parlant_cmd([
        "service", "create", 
        "--name", "AzadeaRAG",
        "--kind", "openapi",
        "--url", "http://localhost:8061",
        "--source", "http://localhost:8061/openapi.json" 
    ])
    
    # 3. Add Guidelines
    print("\n3. Defining Behavioral Guidelines...")
    
    tool_id = "AzadeaRAG:ask_knowledge_base"
    
    guidelines = [
        {
            "condition": "User asks a broad question about policies (e.g. 'leave', 'insurance') without specifying context (Country, Role)",
            "action": "Ask clarifying questions to determine the User's Country (e.g. 'Are you in UAE or Lebanon?') and Role."
        },
        {
            "condition": "User provides specific context (Country, Role) and a question",
            "action": f"Call tool '{tool_id}' to fetch the answer."
        },
        {
            "condition": f"The tool '{tool_id}' returns a response",
            "action": "Present the tool's response to the user clearly."
        }
    ]
    
    
    # Load extracted clustered guidelines if available
    inferred_file = "clustered_guidelines.json"
    if os.path.exists(inferred_file):
        print(f"\n3a. Loading clustered guidelines from {inferred_file}...")
        with open(inferred_file, "r") as f:
            inferred = json.load(f)
            # Load top 15 clusters to demonstrate coverage
            for ig in inferred[:15]:
                 guidelines.append({
                     "condition": ig["condition"],
                     "action": ig["action"]
                 })
    
    for g in guidelines:
        args = [
            "guideline", "create",
            "--condition", g["condition"],
            "--action", g["action"]
        ]
        # Only inject tool_id if the action explicitly mentions calling the tool
        # Wait, I should probably force tool association for fetch actions.
        # But for now, let's just rely on Parlant's inference or explicit tool-id usage if I can detect it.
        # Actually, let's blindly try to associate if 'fetch' or 'call' is in action.
        if "fetch" in g["action"].lower() or "call" in g["action"].lower():
             args.extend(["--tool-id", tool_id])
             
        run_parlant_cmd(args)

    print("\n--- Setup Complete ---")
    print("You can now interact with the agent via Parlant API or CLI.")

if __name__ == "__main__":
    setup_parlant_poc()
