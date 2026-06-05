#!/usr/bin/env python3
"""
RAG System Test Runner
Executes test cases from test_questions.json and validates responses
"""

import json
import requests
import time
import uuid
from datetime import datetime
from pathlib import Path

API_URL = "http://localhost:8060/query"
TEST_FILE = Path(__file__).parent / "test_questions.json"
RESULTS_FILE = Path(__file__).parent / "test_results.json"

def load_tests():
    with open(TEST_FILE, 'r') as f:
        return json.load(f)

def run_single_query(query: str, user_id: str) -> dict:
    """Execute a single query and return the response"""
    try:
        response = requests.post(
            API_URL,
            json={"query": query, "user_id": user_id},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def validate_classification(expected: str, actual: str) -> bool:
    """Check if classification matches expected"""
    return expected.upper() == actual.upper()

def validate_content(response_text: str, expected_contains: list) -> dict:
    """Check if response contains expected content"""
    found = []
    missing = []
    for item in expected_contains:
        if item.lower() in response_text.lower():
            found.append(item)
        else:
            missing.append(item)
    return {
        "found": found,
        "missing": missing,
        "score": len(found) / len(expected_contains) if expected_contains else 1.0
    }

def run_test_case(test_case: dict) -> dict:
    """Run a single test case (potentially multi-turn)"""
    test_id = test_case["id"]
    category = test_case["category"]
    topic = test_case.get("topic", "")
    user_id = f"test_user_{test_id}_{uuid.uuid4().hex[:6]}"
    
    result = {
        "test_id": test_id,
        "category": category,
        "topic": topic,
        "turns": [],
        "passed": True,
        "errors": []
    }
    
    for i, turn in enumerate(test_case["turns"]):
        query = turn["query"]
        expected_classification = turn.get("expected_classification", "")
        expected_contains = test_case.get("expected_answer_contains", [])
        
        print(f"  Turn {i+1}: {query[:50]}...")
        
        start_time = time.time()
        response = run_single_query(query, user_id)
        elapsed = time.time() - start_time
        
        if "error" in response:
            result["passed"] = False
            result["errors"].append(f"Turn {i+1}: API Error - {response['error']}")
            result["turns"].append({
                "query": query,
                "error": response["error"],
                "elapsed_sec": elapsed
            })
            continue
        
        actual_classification = response.get("metadata", {}).get("complexity", "UNKNOWN")
        response_text = response.get("response", "")
        sources = response.get("metadata", {}).get("sources", [])
        
        # Validate classification
        classification_match = validate_classification(expected_classification, actual_classification)
        
        # Validate content (only on last turn if expected_contains is set)
        content_validation = {}
        if i == len(test_case["turns"]) - 1 and expected_contains:
            content_validation = validate_content(response_text, expected_contains)
            if content_validation["missing"]:
                result["errors"].append(f"Missing content: {content_validation['missing']}")
        
        turn_result = {
            "query": query,
            "expected_classification": expected_classification,
            "actual_classification": actual_classification,
            "classification_match": classification_match,
            "response_length": len(response_text),
            "sources_count": len(sources),
            "elapsed_sec": round(elapsed, 2),
            "content_validation": content_validation
        }
        
        if not classification_match:
            result["passed"] = False
            result["errors"].append(f"Turn {i+1}: Classification mismatch - expected {expected_classification}, got {actual_classification}")
        
        result["turns"].append(turn_result)
        
        # Small delay between turns for rate limiting
        time.sleep(0.5)
    
    return result

def run_tests(limit: int = None, categories: list = None):
    """Run all or filtered tests"""
    data = load_tests()
    test_cases = data["test_cases"]
    
    if categories:
        test_cases = [t for t in test_cases if t["category"] in categories]
    
    if limit:
        test_cases = test_cases[:limit]
    
    print(f"\n{'='*60}")
    print(f"RAG System Test Runner")
    print(f"{'='*60}")
    print(f"Total tests to run: {len(test_cases)}")
    print(f"Categories: {categories or 'ALL'}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    results = {
        "run_timestamp": datetime.now().isoformat(),
        "total_tests": len(test_cases),
        "passed": 0,
        "failed": 0,
        "results": []
    }
    
    for i, test_case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Test {test_case['id']}: {test_case['category']} - {test_case.get('topic', 'N/A')}")
        
        result = run_test_case(test_case)
        results["results"].append(result)
        
        if result["passed"]:
            results["passed"] += 1
            print(f"  ✅ PASSED")
        else:
            results["failed"] += 1
            print(f"  ❌ FAILED: {result['errors']}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {results['total_tests']}")
    print(f"Passed: {results['passed']} ({100*results['passed']/results['total_tests']:.1f}%)")
    print(f"Failed: {results['failed']} ({100*results['failed']/results['total_tests']:.1f}%)")
    print(f"{'='*60}\n")
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {RESULTS_FILE}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG System Test Runner")
    parser.add_argument("--limit", type=int, default=10, help="Number of tests to run (default: 10)")
    parser.add_argument("--categories", nargs="+", help="Categories to test (e.g., SIMPLE COMPLEX)")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    limit = None if args.all else args.limit
    run_tests(limit=limit, categories=args.categories)
