
import os
import random
import json
import asyncio
import aiohttp
import time
from datetime import datetime
from pathlib import Path
from simulated_user import SimulatedUser

# Configuration
MD_DIR = "/home/admincsp/multimodal-rag/azadea/md_out_data"
API_URL = "http://localhost:8060/query"
API_RESET_URL = "http://localhost:8060/reset"
OUTPUT_FILE = "simulation_report.json"
NUM_SESSIONS = 100
MAX_TURNS_PER_SESSION = 5

async def run_session(session_id: int, doc_path: str, user_id: str, scenario_type: str = "balanced"):
    print(f"\n--- Session {session_id} (User: {user_id}) [Scenario: {scenario_type}] ---")
    
    # Read doc
    with open(doc_path, "r") as f:
        content = f.read()
    
    # Init Agent
    agent = SimulatedUser(content, os.path.basename(doc_path))
    print(f"Generating goal for {os.path.basename(doc_path)}...")
    try:
        goal = await agent.generate_goal(scenario_type)
        print(f"Goal: {goal.goal_description}")
        print(f"Initial Query: {goal.initial_query}")
    except Exception as e:
        print(f"Failed to generate goal: {e}")
        return None

    # Reset API history for this user
    async with aiohttp.ClientSession() as session:
        await session.post(API_RESET_URL, json={"user_id": user_id})
        
        transcript = []
        final_grade = 0
        status = "incomplete"
        
        # Turn Loop
        current_query = goal.initial_query
        
        for turn in range(MAX_TURNS_PER_SESSION):
            print(f"  Turn {turn+1}: User -> {current_query}")
            start_time = time.time()
            
            # 1. Send to API
            try:
                async with session.post(API_URL, json={"query": current_query, "user_id": user_id}) as resp:
                    if resp.status != 200:
                        print(f"  API Error: {resp.status}")
                        break
                    data = await resp.json()
                    assistant_reply = data.get("response", "")
                    elapsed = time.time() - start_time
                    print(f"  Turn {turn+1}: Assistant -> {assistant_reply[:100]}... ({elapsed:.2f}s)")
            except Exception as e:
                print(f"  Network Error: {e}")
                break
                
            transcript.append({
                "turn": turn,
                "user": current_query,
                "assistant": assistant_reply,
                "latency": elapsed
            })
            
            # 2. Agent Eval
            try:
                eval_result = await agent.process_turn(assistant_reply)
            except Exception as e:
                print(f"  Eval Error: {e}")
                break
                
            if eval_result.is_session_complete:
                final_grade = eval_result.grade
                status = "success" if final_grade >= 7 else "failure"
                print(f"  Session Complete! Grade: {final_grade}/10. Reason: {eval_result.reasoning}")
                break
            else:
                current_query = eval_result.user_response
                if not current_query:
                    print("  Agent returned empty query but session not complete. Aborting.")
                    break
        
        return {
            "session_id": session_id,
            "doc": os.path.basename(doc_path),
            "persona": goal.persona,
            "goal": goal.goal_description,
            "transcript": transcript,
            "grade": final_grade,
            "status": status,
            "scenario": scenario_type,
            "final_reasoning": eval_result.reasoning if 'eval_result' in locals() else "Terminated early"
        }

async def main():
    # 1. Select Docs
    all_files = list(Path(MD_DIR).glob("*.md"))
    # Filter for interesting ones (limit to actual policy/workflow docs, avoid tiny fragments)
    valid_files = [f for f in all_files if f.stat().st_size > 2000]
    
    if not valid_files:
        print("No valid markdown files found!")
        return

    selected_docs = random.choices(valid_files, k=NUM_SESSIONS)
    
    results = []
    
    print(f"Starting {NUM_SESSIONS} simulated sessions...")
    
    # Sequential for now to be nice to the rate limits, or small batch
    for i, doc in enumerate(selected_docs):
        # Assign Scenario
        rand_val = random.random()
        if rand_val < 0.2:
            scenario = "image_workflow"
        elif rand_val < 0.4:
            scenario = "dynamic_followup"
        else:
            scenario = "balanced"
            
        user_id = f"sim_user_{i}_{int(time.time())}"
        result = await run_session(i+1, str(doc), user_id, scenario)
        if result:
            results.append(result)
        
        # Save intermediate
        if i % 5 == 0:
             with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

    # Final Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
        
    # Analysis
    grades = [r["grade"] for r in results]
    avg_grade = sum(grades) / len(grades) if grades else 0
    success_count = sum(1 for r in results if r["status"] == "success")
    
    print("\n--- Simulation Report ---")
    print(f"Total Sessions: {len(results)}")
    print(f"Average Grade: {avg_grade:.2f}/10")
    print(f"Success Rate: {success_count/len(results)*100:.1f}%")
    print(f"Report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
