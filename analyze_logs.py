
import json
import re
import numpy as np
from datetime import datetime, timedelta

LOG_FILE = "/home/admincsp/conversational_rag/logs/rag_server.log"
CURRENT_DATE = datetime(2026, 1, 22)
START_DATE = CURRENT_DATE - timedelta(days=4)

def parse_log_line(line):
    # Example line: 
    # 2026-01-20 16:12:54 | INFO | RAG-Server | [c7ffc776] ⏱️ TIMING_PROFILE | {"1_components": 0.0, ...}
    match = re.match(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*TIMING_PROFILE \| ({.*})$", line)
    if not match:
        return None
    
    date_str, time_str, json_str = match.groups()
    log_date = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    
    if log_date < START_DATE:
        return None
        
    try:
        data = json.loads(json_str)
        return data
    except:
        return None

def parse_deep_agent_end(line):
    # Example: 🤖 DEEP_AGENT_END | {"elapsed_sec": 35.851, "complexity": "COMPLEX", "sub_queries": 1, ...}
    if "DEEP_AGENT_END" not in line:
        return None
    
    data = {}
    # Extract Elapsed Time
    match_time = re.search(r'"elapsed_sec":\s*([\d\.]+)', line)
    if match_time:
        data["0_Total_Latency"] = float(match_time.group(1))
        
    # Extract Complexity
    match_complex = re.search(r'"complexity":\s*"([^"]+)"', line)
    if match_complex:
        data["complexity"] = match_complex.group(1) # "SIMPLE" or "COMPLEX"
        
    # Extract Sub-queries
    match_sub = re.search(r'"sub_queries":\s*(\d+)', line)
    if match_sub:
        data["sub_queries"] = int(match_sub.group(1))
        
    return data

def parse_retrieval_line(line):
    # Example: ⏱️ RETRIEVAL_TIMING: embed=1.156s, parallel_search=1.229s, rerank=5.750s, complete_docs=0.002s, total=9.343s
    if "RETRIEVAL_TIMING" not in line:
        return None
    
    match = re.search(r"embed=([\d\.]+)s.*parallel_search=([\d\.]+)s.*rerank=([\d\.]+)s", line)
    if match:
        return {
            "retrieval_embed": float(match.group(1)),
            "retrieval_search": float(match.group(2)),
            "retrieval_rerank": float(match.group(3))
        }
    return None

def analyze_logs():
    # Structure: requests[req_id] = { "complexity":Str, "total":Float, "components":Dict }
    requests = {}
    
    print(f"Analyzing logs from {START_DATE} to {CURRENT_DATE}...")
    
    try:
        with open(LOG_FILE, 'r') as f:
            last_req_id = None
            
            for line in f:
                # 1. Try to find Request ID in the line
                req_match = re.search(r'\[([a-f0-9]+)\]', line)
                if req_match:
                    req_id = req_match.group(1)
                    last_req_id = req_id # Update context
                else:
                    # If no ID in line, use the last seen one (contextual)
                    req_id = last_req_id

                if not req_id:
                    continue
                
                if req_id not in requests:
                    requests[req_id] = {"components": {}}

                # 2. Total Time & Complexity
                deep_end_data = parse_deep_agent_end(line)
                if deep_end_data:
                    if "0_Total_Latency" in deep_end_data:
                        requests[req_id]["total"] = deep_end_data["0_Total_Latency"]
                    if "complexity" in deep_end_data:
                        requests[req_id]["complexity"] = deep_end_data["complexity"]
                    if "sub_queries" in deep_end_data:
                        requests[req_id]["sub_queries"] = deep_end_data["sub_queries"]

                # 3. Component Profile
                timing_data = parse_log_line(line)
                if timing_data:
                    for k, v in timing_data.items():
                        requests[req_id]["components"][k] = float(v)
                
                # 4. Retrieval Details (Often missing ID)
                retrieval_data = parse_retrieval_line(line)
                if retrieval_data:
                    for k, v in retrieval_data.items():
                        # Key collision handling: If multiple retrievals (Correction loop), sum them or keep list? 
                        # For SIMPLE queries there's usually 1. If multiple, we might overwrite. 
                        # Comparison depends on knowing "Reranking" time. Let's start with overwrite or sum.
                        # Simplest: Update/Max.
                        current_val = requests[req_id]["components"].get(k, 0.0)
                        requests[req_id]["components"][k] = max(current_val, float(v))
                        
    except FileNotFoundError:
        print(f"Error: Log file not found at {LOG_FILE}")
        return

    # Filter Valid Requests (must have total time & complexity)
    valid_reqs = [r for r in requests.values() if "total" in r and "complexity" in r]
    
    # Filter SIMPLE queries
    simple_reqs = [r for r in valid_reqs if r["complexity"] == "SIMPLE"]
    if not simple_reqs:
        print("No SIMPLE queries found.")
        return

    # Calculate Median Time for SIMPLE queries
    totals = [r["total"] for r in simple_reqs]
    median_time = np.median(totals)
    
    # Split into Fast and Slow
    fast_simple = [r for r in simple_reqs if r["total"] <= median_time]
    slow_simple = [r for r in simple_reqs if r["total"] > median_time]
    
    with open("analysis_report.txt", "w") as out:
        def log(msg=""):
            print(msg)
            out.write(msg + "\n")

        log(f"Analyzed {len(valid_reqs)} total requests.")
        log(f"Found {len(simple_reqs)} SIMPLE queries.")
        log(f"Median Time for SIMPLE: {median_time:.3f}s")
        log(f"Slow Subset (> Median): {len(slow_simple)} requests")
        log("-" * 60)
        
        # Compare Components: Slow vs All Simple
        components = [
            "1_components", "2_history", "3_graphiti_context", "4_user_profile", 
            "5_topic_detection", "6_general_handler", "7_query_rewrite", 
            "8_langgraph", "retrieval_embed", "retrieval_search", "retrieval_rerank"
        ]
        
        log(f"{'Component':<25} | {'Fast (Mean)':<12} | {'SLOW (Mean)':<12} | {'Diff':<8}")
        log("-" * 65)
        
        for comp in components:
            # Mean for Fast Subgroup
            fast_vals = [r["components"].get(comp, 0.0) for r in fast_simple]
            fast_mean = np.mean(fast_vals) if fast_vals else 0.0
            
            # Mean for Slow Subgroup
            slow_vals = [r["components"].get(comp, 0.0) for r in slow_simple]
            slow_mean = np.mean(slow_vals) if slow_vals else 0.0
            
            diff = slow_mean - fast_mean
            indicator = "🔴" if diff > 1.0 else ("ASK" if diff > 0.5 else "")
            
            log(f"{comp:<25} | {fast_mean:.3f}s      | {slow_mean:.3f}s      | +{diff:.2f}s {indicator}")

        log("\n=== CANDIDATE REQUEST IDs FOR ANALYSIS ===")
        
        # Sort by duration
        simple_reqs.sort(key=lambda x: x["total"], reverse=True)
        complex_reqs = [r for r in valid_reqs if r["complexity"] == "COMPLEX"]
        complex_reqs.sort(key=lambda x: x["total"], reverse=True)
        
        # 1. Slowest Simple
        log("\n[Top 3 Slowest SIMPLE Queries]")
        for i in range(min(3, len(simple_reqs))):
            r = simple_reqs[i]
            log(f"ID: {get_key_from_value(requests, r)} | Time: {r['total']:.2f}s")
            
        # 2. Slowest Complex
        log("\n[Top 3 Slowest COMPLEX Queries]")
        for i in range(min(3, len(complex_reqs))):
            r = complex_reqs[i]
            log(f"ID: {get_key_from_value(requests, r)} | Time: {r['total']:.2f}s | Sub-queries: {r.get('sub_queries', '?')}")

        # 3. Median Simple
        mid = len(simple_reqs) // 2
        r = simple_reqs[mid]
        log(f"\n[Median SIMPLE Query]")
        log(f"ID: {get_key_from_value(requests, r)} | Time: {r['total']:.2f}s")

def get_key_from_value(d, val):
    # Helper to find key (req_id) by value object identity
    for k, v in d.items():
        if v is val:
            return k
    return "UNKNOWN"


if __name__ == "__main__":
    try:
        import numpy
    except ImportError:
        print("Installing numpy...")
        import subprocess
        subprocess.check_call(["pip", "install", "numpy"])
    
    analyze_logs()
