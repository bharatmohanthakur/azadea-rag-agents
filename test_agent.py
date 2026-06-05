#!/usr/bin/env python3
"""
Advanced AI Test Agent for RAG System Evaluation
================================================

This agent tests the RAG system comprehensively by:
1. Loading test questions from CSV (~84 questions)
2. Sending queries to the RAG server
3. Deep evaluation using LLM reasoning
4. Generating detailed debug reports
5. Thinking conversationally and deeply about each response
"""

import csv
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import re

load_dotenv()

# Configuration
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://localhost:8060")
CSV_FILE_PATH = "/home/admincsp/frontend_integration/CSP BrainShift GenAI Chatbot Test Template.xlsx - Chatbot Test Questions (1).csv"
OUTPUT_DIR = "/home/admincsp/frontend_integration/test_reports"
TIMEOUT = 120  # seconds

# Azure OpenAI for evaluation - use same config as RAG server
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
# Use same deployment name as RAG server (AZURE_OPENAI_CHAT_DEPLOYMENT)
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

@dataclass
class TestQuestion:
    """Represents a test question from CSV"""
    category: str
    question: str
    expected_answer: str
    owner: str
    row_number: int

@dataclass
class TestResult:
    """Result of testing a single question"""
    question: TestQuestion
    actual_answer: str
    response_time: float
    status_code: int
    error: Optional[str] = None
    sources: List[str] = None
    confidence: Optional[str] = None
    
    # Deep evaluation results
    accuracy_score: Optional[float] = None
    semantic_similarity: Optional[float] = None
    completeness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    evaluation_reasoning: Optional[str] = None
    improvement_suggestions: Optional[str] = None
    issues_found: List[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.issues_found is None:
            self.issues_found = []

class ConversationalEvaluator:
    """LLM-based evaluator that thinks deeply and conversationally"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
    
    def evaluate_response(
        self, 
        question: str, 
        expected_answer: str, 
        actual_answer: str,
        category: str
    ) -> Dict:
        """
        Deep evaluation using LLM with conversational reasoning.
        Thinks like a human tester would.
        """
        # Detect if expected answer is an instruction/requirement vs actual content
        expected_lower = expected_answer.lower()
        is_instruction = any(keyword in expected_lower for keyword in [
            'should', 'must', 'prefer', 'able to', 'not able', 'chatbot', 'system',
            'provide as', 'format', 'out of context', 'it should', 'should be'
        ])
        
        instruction_guidance = ""
        if is_instruction:
            instruction_guidance = """
**IMPORTANT: The "Expected Answer" appears to be a TEST INSTRUCTION or REQUIREMENT, not actual expected content.**
- If it describes what the system SHOULD do (e.g., "should provide as table", "not able to present data"), evaluate whether the AI followed the instruction.
- If it describes a limitation (e.g., "chatbot is not able to present the data as table"), the AI should acknowledge this limitation.
- If it's a preference (e.g., "i prefer to provide it as a table"), evaluate if the AI provided the information, even if not in the preferred format.
- Extract any actual expected CONTENT from the instruction and evaluate against that.
- Focus on whether the AI answered the USER'S QUESTION correctly, not just whether it followed the instruction format.
"""
        
        prompt = f"""You are an expert QA tester evaluating an AI chatbot's response. Think deeply and conversationally, like a human would when reviewing answers.

**CONTEXT:**
- Question Category: {category}
- User Question: "{question}"
- Expected Answer/Instruction: "{expected_answer}"
- Actual AI Response: "{actual_answer}"
{instruction_guidance}
**YOUR TASK:**
Evaluate the AI's response as if you were a human tester having a conversation with yourself about the quality. Think step by step:

1. **Understanding Check**: Does the AI understand what was asked?
2. **Answer Quality**: Does the AI provide a helpful, accurate answer to the user's question?
3. **Content Evaluation**: If the expected answer contains actual content (not just instructions), does the AI's answer include that content?
4. **Instruction Compliance**: If the expected answer is an instruction/requirement, did the AI follow it appropriately? (But don't penalize heavily if the answer is good but format differs)
5. **Completeness Check**: Does it cover all important points that a user would need?
6. **Relevance Check**: Is it directly answering the question or going off-topic?
7. **Clarity Check**: Is it well-structured and easy to understand?
8. **Missing Information**: What key information is missing?
9. **Extra Information**: Does it add unnecessary or incorrect information?

**CRITICAL EVALUATION PRINCIPLES:**
- **Primary Focus**: Does the AI answer the USER'S QUESTION correctly and helpfully?
- **Instruction vs Content**: If expected answer is an instruction, evaluate whether the AI followed it, BUT prioritize whether the user's question was answered correctly.
- **Format vs Content**: If format is requested (table, points) but content is correct, don't heavily penalize format issues if content is good.
- **Limitations**: If expected answer describes a system limitation, the AI should acknowledge it appropriately.
- **Be Fair**: A good answer that helps the user should score well, even if it doesn't match every detail of the expected answer/instruction.

**THINK DEEPLY:**
- Consider what a real user would find helpful
- Consider if the answer would solve the user's actual problem
- Don't be overly strict about format if content is correct
- Evaluate the answer's usefulness, not just strict adherence to instructions

**OUTPUT FORMAT (JSON):**
{{
    "accuracy_score": 0.0-10.0,
    "semantic_similarity": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "evaluation_reasoning": "Your detailed conversational reasoning, thinking through each aspect step by step. Write as if explaining to a colleague.",
    "improvement_suggestions": "Specific, actionable suggestions for improvement",
    "issues_found": ["issue1", "issue2", ...],
    "key_strengths": ["strength1", "strength2", ...]
}}

**SCORING GUIDELINES:**
- 10: Perfect answer that fully addresses the question
- 8-9: Very good answer with minor issues (format, minor missing details)
- 7: Good answer that addresses the question well (may have format issues or minor gaps)
- 5-6: Acceptable answer but missing some important details or has some inaccuracies
- 3-4: Significant problems - doesn't fully answer the question or has major issues
- 1-2: Major failures - doesn't address the question or is mostly incorrect

Be thorough, fair, and honest. Prioritize whether the user's question was answered correctly over strict format compliance."""

        try:
            response = self.client.chat.completions.create(
                model=AZURE_CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an expert QA tester with deep analytical thinking. Evaluate responses conversationally and thoroughly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Some creativity for deep thinking
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in markdown
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(result_text)
            
            return {
                "accuracy_score": float(evaluation.get("accuracy_score", 5.0)),
                "semantic_similarity": float(evaluation.get("semantic_similarity", 0.5)),
                "completeness_score": float(evaluation.get("completeness_score", 0.5)),
                "relevance_score": float(evaluation.get("relevance_score", 0.5)),
                "evaluation_reasoning": evaluation.get("evaluation_reasoning", ""),
                "improvement_suggestions": evaluation.get("improvement_suggestions", ""),
                "issues_found": evaluation.get("issues_found", []),
                "key_strengths": evaluation.get("key_strengths", [])
            }
            
        except Exception as e:
            print(f"⚠️  Evaluation error: {e}")
            return {
                "accuracy_score": 5.0,
                "semantic_similarity": 0.5,
                "completeness_score": 0.5,
                "relevance_score": 0.5,
                "evaluation_reasoning": f"Evaluation failed: {str(e)}",
                "improvement_suggestions": "Fix evaluation system",
                "issues_found": ["Evaluation system error"],
                "key_strengths": []
            }

class RAGTestAgent:
    """Main test agent that orchestrates testing"""
    
    def __init__(self):
        self.evaluator = ConversationalEvaluator()
        self.results: List[TestResult] = []
        self.stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "avg_accuracy": 0.0,
            "avg_response_time": 0.0
        }
    
    def load_questions(self, only_with_expected: bool = True) -> List[TestQuestion]:
        """Load test questions from CSV
        
        Args:
            only_with_expected: If True, only load questions that have expected answers (default: True)
        """
        questions = []
        
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=2):  # Start at 2 (row 1 is header)
                question = row.get('User Question', '').strip()
                if not question or question.lower() in ['', 'n/a', 'none']:
                    continue
                
                # Use only "Expected Answer " column (not "AI Chatbot Answer" which is the actual response)
                expected_answer = row.get('Expected Answer ', '').strip()
                
                # Skip questions without expected answers if filtering is enabled
                if only_with_expected:
                    if not expected_answer or expected_answer.lower() in ['', 'n/a', 'none']:
                        continue  # Skip this question
                
                questions.append(TestQuestion(
                    category=row.get('Question Category', 'Unknown').strip(),
                    question=question,
                    expected_answer=expected_answer,
                    owner=row.get('Testing scenario owner', 'Unknown').strip(),
                    row_number=idx
                ))
        
        return questions
    
    def query_rag_server(self, question: str, user_id: str = "test_agent") -> Tuple[Dict, float, int]:
        """Send query to RAG server and return response"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{RAG_SERVER_URL}/query",
                json={"query": question, "user_id": user_id},
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                # Handle both 'response' and 'answer' fields
                answer = data.get("response", data.get("answer", ""))
                metadata = data.get("metadata", {})
                sources = metadata.get("sources", data.get("sources", []))
                # Extract source names
                source_names = []
                if sources:
                    for s in sources:
                        if isinstance(s, dict):
                            source_names.append(s.get("source", s.get("name", "")))
                        else:
                            source_names.append(str(s))
                
                # Extract confidence from response text or metadata
                confidence = "unknown"
                if "confidence" in data:
                    confidence = data["confidence"]
                elif "Confidence:" in answer:
                    # Try to extract from answer text
                    conf_match = re.search(r'Confidence[:\s]+(\w+)', answer, re.IGNORECASE)
                    if conf_match:
                        confidence = conf_match.group(1)
                
                return {
                    "answer": answer,
                    "sources": source_names,
                    "confidence": confidence,
                    "metadata": metadata
                }, elapsed, 200
            else:
                return {
                    "answer": "",
                    "sources": [],
                    "error": f"HTTP {response.status_code}: {response.text[:200]}"
                }, elapsed, response.status_code
                
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            return {
                "answer": "",
                "sources": [],
                "error": f"Request timeout after {TIMEOUT}s"
            }, elapsed, 0
            
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "answer": "",
                "sources": [],
                "error": str(e)
            }, elapsed, 0
    
    def test_question(self, test_q: TestQuestion, verbose: bool = True) -> TestResult:
        """Test a single question"""
        if verbose:
            print(f"\n{'='*80}")
            print(f"📋 Testing Question #{test_q.row_number}")
            print(f"Category: {test_q.category}")
            print(f"Question: {test_q.question[:100]}...")
            print(f"{'='*80}")
        
        # Query RAG server
        response_data, response_time, status_code = self.query_rag_server(test_q.question)
        
        actual_answer = response_data.get("answer", "")
        # Clean HTML tags from answer for better evaluation
        if actual_answer:
            actual_answer = re.sub(r'<[^>]+>', '', actual_answer)  # Remove HTML tags
            actual_answer = re.sub(r'\s+', ' ', actual_answer)  # Normalize whitespace
            actual_answer = actual_answer.strip()
        
        error = response_data.get("error")
        sources = response_data.get("sources", [])
        confidence = response_data.get("confidence")
        
        if verbose:
            print(f"⏱️  Response time: {response_time:.2f}s")
            if error:
                print(f"❌ Error: {error}")
            else:
                print(f"✅ Got response ({len(actual_answer)} chars)")
                print(f"📚 Sources: {len(sources)}")
                if confidence:
                    print(f"🎯 Confidence: {confidence}")
        
        # Create result
        result = TestResult(
            question=test_q,
            actual_answer=actual_answer,
            response_time=response_time,
            status_code=status_code,
            error=error,
            sources=sources,
            confidence=confidence
        )
        
        # Deep evaluation if we got a response
        if not error and actual_answer and test_q.expected_answer:
            if verbose:
                print(f"🧠 Deep evaluation in progress...")
            
            evaluation = self.evaluator.evaluate_response(
                question=test_q.question,
                expected_answer=test_q.expected_answer,
                actual_answer=actual_answer,
                category=test_q.category
            )
            
            result.accuracy_score = evaluation["accuracy_score"]
            result.semantic_similarity = evaluation["semantic_similarity"]
            result.completeness_score = evaluation["completeness_score"]
            result.relevance_score = evaluation["relevance_score"]
            result.evaluation_reasoning = evaluation["evaluation_reasoning"]
            result.improvement_suggestions = evaluation["improvement_suggestions"]
            result.issues_found = evaluation["issues_found"]
            
            if verbose:
                print(f"📊 Accuracy Score: {result.accuracy_score:.1f}/10.0")
                print(f"📈 Semantic Similarity: {result.semantic_similarity:.2f}")
                print(f"✅ Completeness: {result.completeness_score:.2f}")
                print(f"🎯 Relevance: {result.relevance_score:.2f}")
                if result.issues_found:
                    print(f"⚠️  Issues: {', '.join(result.issues_found[:3])}")
        
        self.results.append(result)
        return result
    
    def run_all_tests(self, limit: Optional[int] = None, verbose: bool = True, only_with_expected: bool = True) -> None:
        """Run all tests
        
        Args:
            limit: Limit number of questions to test
            verbose: Show detailed output
            only_with_expected: Only test questions that have expected answers (default: True)
        """
        # Load questions with filtering applied at load time
        questions = self.load_questions(only_with_expected=only_with_expected)
        
        if verbose and only_with_expected:
            print(f"📋 Loaded {len(questions)} questions with expected answers (excluded {84 - len(questions)} without expected answers)")
        
        if limit:
            questions = questions[:limit]
        
        total = len(questions)
        self.stats["total"] = total
        
        print(f"\n🚀 Starting comprehensive test run")
        print(f"📊 Total questions: {total}")
        print(f"🌐 RAG Server: {RAG_SERVER_URL}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Estimated time: ~{total * 30 // 60} minutes (assuming ~30s per question)")
        print(f"\n{'='*80}\n")
        
        for idx, question in enumerate(questions, 1):
            try:
                result = self.test_question(question, verbose=verbose)
                
                # Update stats
                if result.error:
                    self.stats["errors"] += 1
                elif result.accuracy_score:
                    if result.accuracy_score >= 7.0:
                        self.stats["passed"] += 1
                    else:
                        self.stats["failed"] += 1
                
                # Progress update
                if verbose and idx % 10 == 0:
                    print(f"\n📈 Progress: {idx}/{total} ({idx*100//total}%)")
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Test interrupted by user at question {idx}/{total}")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error on question {idx}: {e}")
                self.stats["errors"] += 1
                continue
        
        # Calculate final stats
        self._calculate_stats()
        
        # Generate report
        self.generate_report()
    
    def _calculate_stats(self):
        """Calculate statistics"""
        if not self.results:
            return
        
        # Average accuracy
        accuracy_scores = [r.accuracy_score for r in self.results if r.accuracy_score]
        if accuracy_scores:
            self.stats["avg_accuracy"] = sum(accuracy_scores) / len(accuracy_scores)
        
        # Average response time
        response_times = [r.response_time for r in self.results]
        if response_times:
            self.stats["avg_response_time"] = sum(response_times) / len(response_times)
    
    def generate_report(self):
        """Generate detailed debug report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(OUTPUT_DIR)
        report_dir.mkdir(exist_ok=True)
        
        # Summary report
        summary_file = report_dir / f"test_summary_{timestamp}.json"
        detailed_file = report_dir / f"test_detailed_{timestamp}.json"
        markdown_file = report_dir / f"test_report_{timestamp}.md"
        
        # Save JSON reports
        summary_data = {
            "timestamp": timestamp,
            "stats": self.stats,
            "total_questions": len(self.results),
            "categories": {}
        }
        
        # Group by category
        for result in self.results:
            cat = result.question.category
            if cat not in summary_data["categories"]:
                summary_data["categories"][cat] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "avg_accuracy": 0.0
                }
            
            summary_data["categories"][cat]["total"] += 1
            if result.accuracy_score:
                if result.accuracy_score >= 7.0:
                    summary_data["categories"][cat]["passed"] += 1
                else:
                    summary_data["categories"][cat]["failed"] += 1
        
        # Calculate category averages
        for cat in summary_data["categories"]:
            cat_results = [r for r in self.results if r.question.category == cat and r.accuracy_score]
            if cat_results:
                summary_data["categories"][cat]["avg_accuracy"] = sum(r.accuracy_score for r in cat_results) / len(cat_results)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Detailed results
        detailed_data = {
            "timestamp": timestamp,
            "stats": self.stats,
            "results": [asdict(r) for r in self.results]
        }
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        # Markdown report
        self._generate_markdown_report(markdown_file, summary_data)
        
        print(f"\n{'='*80}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Questions: {self.stats['total']}")
        print(f"Passed (≥7.0): {self.stats['passed']}")
        print(f"Failed (<7.0): {self.stats['failed']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Average Accuracy: {self.stats['avg_accuracy']:.2f}/10.0")
        print(f"Average Response Time: {self.stats['avg_response_time']:.2f}s")
        print(f"\n📁 Reports saved to:")
        print(f"   - Summary: {summary_file}")
        print(f"   - Detailed: {detailed_file}")
        print(f"   - Markdown: {markdown_file}")
        print(f"{'='*80}\n")
    
    def _generate_markdown_report(self, filepath: Path, summary_data: Dict):
        """Generate human-readable markdown report"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# RAG System Test Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write(f"## Executive Summary\n\n")
            f.write(f"- **Total Questions Tested:** {self.stats['total']}\n")
            f.write(f"- **Passed (≥7.0):** {self.stats['passed']}\n")
            f.write(f"- **Failed (<7.0):** {self.stats['failed']}\n")
            f.write(f"- **Errors:** {self.stats['errors']}\n")
            f.write(f"- **Average Accuracy:** {self.stats['avg_accuracy']:.2f}/10.0\n")
            f.write(f"- **Average Response Time:** {self.stats['avg_response_time']:.2f}s\n\n")
            
            # Category Breakdown
            f.write(f"## Category Breakdown\n\n")
            f.write(f"| Category | Total | Passed | Failed | Avg Accuracy |\n")
            f.write(f"|----------|-------|--------|--------|--------------|\n")
            for cat, data in summary_data["categories"].items():
                f.write(f"| {cat} | {data['total']} | {data['passed']} | {data['failed']} | {data['avg_accuracy']:.2f} |\n")
            f.write(f"\n")
            
            # Detailed Results
            f.write(f"## Detailed Results\n\n")
            for idx, result in enumerate(self.results, 1):
                f.write(f"### Question #{result.question.row_number}: {result.question.category}\n\n")
                f.write(f"**Question:** {result.question.question}\n\n")
                f.write(f"**Expected Answer:** {result.question.expected_answer[:200]}...\n\n")
                f.write(f"**Actual Answer:** {result.actual_answer[:300]}...\n\n")
                
                if result.error:
                    f.write(f"**❌ Error:** {result.error}\n\n")
                else:
                    f.write(f"**Metrics:**\n")
                    if result.accuracy_score is not None:
                        f.write(f"- Accuracy: {result.accuracy_score:.1f}/10.0\n")
                    if result.semantic_similarity is not None:
                        f.write(f"- Semantic Similarity: {result.semantic_similarity:.2f}\n")
                    if result.completeness_score is not None:
                        f.write(f"- Completeness: {result.completeness_score:.2f}\n")
                    if result.relevance_score is not None:
                        f.write(f"- Relevance: {result.relevance_score:.2f}\n")
                    f.write(f"- Response Time: {result.response_time:.2f}s\n")
                    f.write(f"- Sources: {len(result.sources)}\n\n")
                    
                    if result.evaluation_reasoning:
                        f.write(f"**Evaluation Reasoning:**\n{result.evaluation_reasoning}\n\n")
                    
                    if result.issues_found:
                        f.write(f"**Issues Found:**\n")
                        for issue in result.issues_found:
                            f.write(f"- {issue}\n")
                        f.write(f"\n")
                    
                    if result.improvement_suggestions:
                        f.write(f"**Improvement Suggestions:**\n{result.improvement_suggestions}\n\n")
                
                f.write(f"---\n\n")

def main():
    """Main entry point"""
    import argparse
    
    global RAG_SERVER_URL
    
    parser = argparse.ArgumentParser(description="RAG System Test Agent")
    parser.add_argument("--limit", type=int, help="Limit number of questions to test")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--server", type=str, help="RAG server URL", default=RAG_SERVER_URL)
    
    args = parser.parse_args()
    
    if args.server:
        RAG_SERVER_URL = args.server
    
    # Check server health
    try:
        response = requests.get(f"{RAG_SERVER_URL}/health", timeout=10)
        if response.status_code != 200:
            print(f"⚠️  Warning: Server health check returned {response.status_code}")
        else:
            print(f"✅ RAG server is healthy and ready")
    except Exception as e:
        print(f"❌ Cannot connect to RAG server at {RAG_SERVER_URL}")
        print(f"   Error: {e}")
        print(f"\n   Please ensure the server is running.")
        return
    
    agent = RAGTestAgent()
    agent.run_all_tests(limit=args.limit, verbose=not args.quiet)

if __name__ == "__main__":
    main()
