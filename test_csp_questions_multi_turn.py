#!/usr/bin/env python3
"""
Comprehensive test suite for CSP BrainShift questions with multi-turn conversation support.
Handles clarification questions and captures all conversation turns.
Uses LLM-based agent for intelligent clarification responses.
"""

import os
import csv
import json
import time
import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
from dotenv import load_dotenv
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8069/query"
TIMEOUT = 60  # 60 seconds timeout for API requests (queries can take 30-40s with reranking + corrective RAG)

# Azure OpenAI configuration for LLM agent
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")


class LLMAgent:
    """LLM-based agent that intelligently answers clarification questions."""
    
    def __init__(self, expected_answer: str = None, original_question: str = None):
        """
        Initialize LLM agent.
        
        Args:
            expected_answer: The expected answer to provide context
            original_question: The original user question
        """
        self.expected_answer = expected_answer
        self.original_question = original_question
        self.conversation_history = []
        
        # Initialize Azure OpenAI client
        if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
            self.llm_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version="2024-02-01"
            )
        else:
            logger.warning("Azure OpenAI credentials not found. Agent will use fallback logic.")
            self.llm_client = None
    
    def answer_clarification(self, clarification_question: str) -> str:
        """
        Generate an intelligent answer to a clarification question using LLM.
        
        Args:
            clarification_question: The clarification question from the system
            
        Returns:
            Answer to the clarification question
        """
        if not self.llm_client:
            # Fallback to simple extraction
            return self._fallback_answer(clarification_question)
        
        try:
            # Build context for LLM
            context_parts = []
            
            if self.original_question:
                context_parts.append(f"Original Question: {self.original_question}")
            
            if self.expected_answer:
                context_parts.append(f"Expected Answer Context: {self.expected_answer[:1000]}")  # Limit length
            
            if self.conversation_history:
                context_parts.append("Previous Conversation:")
                for turn in self.conversation_history[-3:]:  # Last 3 turns
                    context_parts.append(f"- {turn.get('question', '')}: {turn.get('response', '')[:200]}")
            
            context = "\n".join(context_parts)
            
            prompt = f"""You are a helpful employee at Azadea company asking HR questions. The HR chatbot has asked you a clarification question. 

Based on the context below, provide a natural, concise answer to the clarification question. Your answer should be realistic and help the chatbot provide the best response.

Context:
{context}

Clarification Question: {clarification_question}

Provide a brief, natural answer (1-2 sentences maximum) that would help the chatbot understand what you need:"""

            response = self.llm_client.chat.completions.create(
                model=AZURE_CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a helpful employee at Azadea. Answer clarification questions naturally and concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            if answer.startswith('"') and answer.endswith('"'):
                answer = answer[1:-1]
            
            logger.info(f"  → LLM Agent answered: {answer[:80]}...")
            return answer
            
        except Exception as e:
            logger.warning(f"LLM agent error: {e}, using fallback")
            return self._fallback_answer(clarification_question)
    
    def _fallback_answer(self, clarification_question: str) -> str:
        """Simple fallback when LLM is not available."""
        question_lower = clarification_question.lower()
        context = self.expected_answer.lower() if self.expected_answer else ""
        
        # Simple keyword matching
        if "country" in question_lower:
            countries = ["lebanon", "uae", "egypt", "saudi", "kuwait", "qatar", "jordan"]
            for country in countries:
                if country in context:
                    return country.title()
            return "Lebanon"
        
        elif "position" in question_lower or "role" in question_lower:
            positions = ["manager", "employee", "director", "supervisor"]
            for pos in positions:
                if pos in context:
                    return pos.title()
            return "Employee"
        
        elif "yes" in question_lower or "no" in question_lower:
            if any(word in context for word in ["yes", "eligible", "can", "allowed"]):
                return "yes"
            return "no"
        
        return "I need more information"
    
    def add_conversation_turn(self, question: str, response: str):
        """Add a turn to conversation history."""
        self.conversation_history.append({
            "question": question,
            "response": response
        })
    
    def check_if_clarification_needed(self, api_response: str, original_question: str) -> Tuple[bool, str]:
        """
        Use LLM to check if clarification is needed and generate answer if needed.
        
        Args:
            api_response: The response from the API
            original_question: The original question asked
            
        Returns:
            Tuple of (needs_clarification: bool, answer_to_use: str)
        """
        if not self.llm_client:
            # Fallback: check for common clarification patterns
            response_lower = api_response.lower()
            clarification_indicators = [
                "need more information",
                "which country",
                "what position",
                "which type",
                "please specify",
                "could you clarify",
                "i need to know"
            ]
            needs_clarification = any(indicator in response_lower for indicator in clarification_indicators)
            return needs_clarification, self._fallback_answer(api_response) if needs_clarification else ""
        
        try:
            # Build context
            context_parts = []
            if self.original_question:
                context_parts.append(f"Original Question: {self.original_question}")
            if self.expected_answer:
                context_parts.append(f"Expected Answer Context: {self.expected_answer[:800]}")
            if self.conversation_history:
                context_parts.append("Previous Conversation:")
                for turn in self.conversation_history[-2:]:
                    context_parts.append(f"- Q: {turn.get('question', '')[:100]}")
                    context_parts.append(f"- A: {turn.get('response', '')[:200]}")
            
            context = "\n".join(context_parts)
            
            prompt = f"""You are testing an HR chatbot system. The chatbot just responded to a question.

Original Question: {original_question}

Chatbot Response:
{api_response}

Context (for generating appropriate answers):
{context}

Analyze the chatbot's response and determine:
1. Does the chatbot need clarification? (Look for questions like "Which country?", "What position?", "Please specify", etc.)
2. If clarification is needed, what should be the answer to help the chatbot provide the best response?

Respond in JSON format:
{{
    "needs_clarification": true/false,
    "reason": "brief explanation",
    "answer": "the answer to provide if clarification is needed, or empty string if not needed"
}}

Be concise. If clarification is needed, provide a natural 1-2 word or short phrase answer based on the expected answer context."""

            response = self.llm_client.chat.completions.create(
                model=AZURE_CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a test agent analyzing chatbot responses. Respond only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            needs_clarification = result.get("needs_clarification", False)
            answer = result.get("answer", "").strip()
            
            logger.info(f"  → LLM Analysis: needs_clarification={needs_clarification}, answer={answer[:60]}...")
            
            return needs_clarification, answer
            
        except Exception as e:
            logger.warning(f"LLM check error: {e}, using fallback")
            # Fallback check
            response_lower = api_response.lower()
            needs_clarification = any(phrase in response_lower for phrase in [
                "which", "what", "please specify", "need more", "could you clarify"
            ])
            return needs_clarification, self._fallback_answer(api_response) if needs_clarification else ""


class TestRunner:
    """Runs tests against the RAG API with multi-turn conversation support."""
    
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.results = []
        
        # Initialize LLM client for evaluation
        if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
            self.llm_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version="2024-02-01"
            )
        else:
            logger.warning("Azure OpenAI credentials not found. Evaluation will use fallback.")
            self.llm_client = None
    
    def query_api(self, question: str, user_id: str = "test_user") -> Dict:
        """Send query to API and get response."""
        try:
            response = requests.post(
                self.api_url,
                json={"query": question, "user_id": user_id},
                timeout=TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            # Normalize response format
            if "response" in result:
                result["final_answer"] = result["response"]  # Add for compatibility
            return result
        except requests.exceptions.Timeout:
            logger.error(f"Timeout for question: {question}")
            return {"error": "Request timeout", "response": "Timeout occurred", "final_answer": "Timeout occurred"}
        except Exception as e:
            logger.error(f"Error querying API: {e}")
            return {"error": str(e), "response": f"Error: {str(e)}", "final_answer": f"Error: {str(e)}"}
    
    def run_multi_turn_test(
        self,
        question: str,
        expected_answer: str,
        category: str,
        owner: str,
        user_id: str = None
    ) -> Dict:
        """
        Run a test with multi-turn conversation support.
        
        Args:
            question: Initial question
            expected_answer: Expected final answer
            category: Question category
            owner: Test scenario owner
            user_id: User ID for conversation tracking
            
        Returns:
            Test result dictionary
        """
        if user_id is None:
            user_id = f"test_{category}_{int(time.time())}"
        
        agent = LLMAgent(expected_answer=expected_answer, original_question=question)
        conversation_turns = []
        max_turns = 3  # Maximum 3 turns total (1 initial + 2 clarifications)
        turn_count = 0
        
        current_question = question
        final_answer = None
        sources = []
        errors = []
        
        logger.info(f"Testing: {question[:50]}...")
        
        while turn_count < max_turns:
            turn_count += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"Turn {turn_count}/{max_turns}")
            logger.info(f"{'='*80}")
            logger.info(f"Question: {current_question}")
            logger.info(f"{'-'*80}")
            
            # Query API
            response = self.query_api(current_question, user_id)
            
            # Extract response text
            response_text = response.get("response", response.get("final_answer", ""))
            
            # Print response in terminal
            logger.info(f"\nResponse:")
            logger.info(f"{'-'*80}")
            # Print response with proper formatting (limit to reasonable length for terminal)
            if len(response_text) > 500:
                logger.info(f"{response_text[:500]}...")
                logger.info(f"[Response truncated - full length: {len(response_text)} chars]")
            else:
                logger.info(response_text)
            logger.info(f"{'-'*80}")
            
            # Capture turn
            turn_data = {
                "turn": turn_count,
                "question": current_question,
                "response": response_text,
                "awaiting_clarification": response.get("awaiting_clarification", False),
                "clarifying_questions": response.get("clarifying_questions", []),
                "sources": response.get("sources", []),
                "error": response.get("error")
            }
            conversation_turns.append(turn_data)
            
            if response.get("error"):
                errors.append(response["error"])
                break
            
            # Extract response text
            response_text = response.get("response", response.get("final_answer", ""))
            
            # Add turn to agent history
            agent.add_conversation_turn(current_question, response_text)
            
            # Use LLM to check if clarification is needed
            logger.info(f"\nAnalyzing response for clarification needs...")
            needs_clarification, clarification_answer = agent.check_if_clarification_needed(
                response_text, 
                question  # Original question
            )
            
            if needs_clarification and clarification_answer:
                logger.info(f"\n{'='*80}")
                logger.info(f"✓ Clarification needed!")
                logger.info(f"LLM generated answer: {clarification_answer}")
                logger.info(f"{'='*80}\n")
                # Use LLM's answer as next question
                current_question = clarification_answer
            else:
                # No clarification needed - we have final answer
                logger.info(f"\n{'='*80}")
                logger.info(f"✓ No clarification needed - Final answer received")
                logger.info(f"{'='*80}\n")
                final_answer = response_text
                sources = response.get("metadata", {}).get("sources", [])
                break
        
        if not final_answer and conversation_turns:
            # Use last response as final answer
            final_answer = conversation_turns[-1]["response"]
        
        # Print final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total Turns: {turn_count}")
        logger.info(f"Final Answer Length: {len(final_answer)} chars")
        logger.info(f"Sources Found: {len(sources)}")
        logger.info(f"{'='*80}\n")
        
        # Calculate accuracy score using LLM judge
        logger.info(f"Evaluating answer accuracy with LLM judge...")
        accuracy_score = self.evaluate_answer(final_answer, expected_answer, question)
        logger.info(f"Accuracy Score: {accuracy_score:.1f}/10\n")
        
        return {
            "category": category,
            "question": question,
            "expected_answer": expected_answer,
            "final_answer": final_answer,
            "accuracy_score": accuracy_score,
            "turns": turn_count,
            "conversation_turns": conversation_turns,
            "sources_count": len(sources),
            "errors": errors,
            "owner": owner
        }
    
    def evaluate_answer(self, actual: str, expected: str, question: str = None) -> float:
        """
        Evaluate answer accuracy on a scale of 1-10 using LLM as judge.
        
        Args:
            actual: Actual answer from system
            expected: Expected answer
            question: Original question (for context)
            
        Returns:
            Score from 1-10
        """
        if not actual or not expected:
            return 1.0
        
        # Use LLM to evaluate if available
        if self.llm_client:
            return self._llm_evaluate_answer(actual, expected, question)
        else:
            # Fallback to keyword-based evaluation
            return self._keyword_evaluate_answer(actual, expected)
    
    def _llm_evaluate_answer(self, actual: str, expected: str, question: str = None) -> float:
        """Use LLM to evaluate answer relevancy and accuracy."""
        try:
            prompt = f"""You are an expert evaluator assessing the quality of an AI chatbot's answer.

Original Question: {question if question else "N/A"}

Expected Answer (reference):
{expected[:1500]}

Actual Answer (to evaluate):
{actual[:1500]}

Evaluate the actual answer on these criteria:
1. **Relevancy**: Does it address the question?
2. **Completeness**: Does it cover the key points from the expected answer?
3. **Accuracy**: Are the facts correct?
4. **Clarity**: Is it well-structured and clear?

Provide a score from 1-10 where:
- 10: Perfect match, all key points covered accurately
- 8-9: Very good, minor details missing
- 6-7: Good, covers main points but missing some details
- 4-5: Fair, addresses question but incomplete or partially inaccurate
- 2-3: Poor, misses key points or has significant inaccuracies
- 1: Very poor, doesn't address the question or is mostly incorrect

Respond in JSON format:
{{
    "score": 1-10,
    "reasoning": "brief explanation of the score",
    "key_missing_points": ["point 1", "point 2"] or [],
    "inaccuracies": ["issue 1", "issue 2"] or []
}}"""

            response = self.llm_client.chat.completions.create(
                model=AZURE_CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Provide accurate, fair assessments in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            score = float(result.get("score", 5.0))
            
            # Clamp to 1-10
            score = max(1.0, min(10.0, score))
            
            reasoning = result.get("reasoning", "")
            logger.info(f"  → LLM Evaluation: {score:.1f}/10 - {reasoning[:60]}...")
            
            return score
            
        except Exception as e:
            logger.warning(f"LLM evaluation error: {e}, using fallback")
            return self._keyword_evaluate_answer(actual, expected)
    
    def _keyword_evaluate_answer(self, actual: str, expected: str) -> float:
        """Fallback keyword-based evaluation."""
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Extract key entities and facts from expected answer
        expected_keywords = set()
        for word in expected_lower.split():
            if len(word) > 4:  # Only meaningful words
                expected_keywords.add(word)
        
        # Check how many keywords are present in actual answer
        found_keywords = sum(1 for kw in expected_keywords if kw in actual_lower)
        
        if len(expected_keywords) == 0:
            return 5.0  # Can't evaluate
        
        keyword_coverage = found_keywords / len(expected_keywords)
        
        # Check for key phrases
        key_phrases = [
            "azadea", "employee", "policy", "leave", "insurance",
            "commission", "bonus", "relocation", "maternity"
        ]
        phrase_matches = sum(1 for phrase in key_phrases if phrase in actual_lower and phrase in expected_lower)
        
        # Calculate score
        base_score = keyword_coverage * 7.0  # Up to 7 points for keyword coverage
        phrase_bonus = min(phrase_matches * 0.5, 2.0)  # Up to 2 points for key phrases
        length_penalty = 0.0
        if len(actual) < len(expected) * 0.3:  # Answer too short
            length_penalty = -1.0
        
        score = base_score + phrase_bonus + length_penalty + 1.0  # Base score of 1
        return max(1.0, min(10.0, score))  # Clamp to 1-10
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to CSV file."""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Question Category",
                "User Question",
                "Expected Answer",
                "AI Chatbot Answer",
                "Accuracy Score (1-10)",
                "Number of Turns",
                "Conversation Turns (JSON)",
                "Sources Count",
                "Errors",
                "Improvement Suggested",
                "Testing scenario owner"
            ])
            
            # Write results
            for result in results:
                # Format conversation turns as JSON string
                turns_json = json.dumps(result["conversation_turns"], ensure_ascii=False)
                
                writer.writerow([
                    result["category"],
                    result["question"],
                    result["expected_answer"],
                    result["final_answer"],
                    f"{result['accuracy_score']:.1f}",
                    result["turns"],
                    turns_json,
                    result["sources_count"],
                    "; ".join(result["errors"]) if result["errors"] else "",
                    "",  # Improvement suggested - to be filled manually
                    result["owner"]
                ])
        
        logger.info(f"Results saved to {output_file}")


def parse_csv(input_file: str) -> List[Dict]:
    """Parse the input CSV file."""
    questions = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            question = row.get("User Question", "").strip()
            expected = row.get("Expected Answer ", "").strip()  # Note the space
            category = row.get("Question Category", "").strip()
            owner = row.get("Testing scenario owner", "").strip()
            
            # Skip empty rows
            if not question:
                continue
            
            # Clean expected answer (remove quotes and newlines in CSV format)
            if expected.startswith('"') and expected.endswith('"'):
                expected = expected[1:-1]
            expected = expected.replace('""', '"')  # Unescape quotes
            
            questions.append({
                "category": category,
                "question": question,
                "expected_answer": expected,
                "owner": owner
            })
    
    return questions


def main():
    """Main test execution."""
    input_file = "CSP BrainShift GenAI Chatbot Test Template.xlsx - Chatbot Test Questions.csv"
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    logger.info("Starting CSP BrainShift Test Suite")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"API URL: {API_URL}")
    
    # Parse questions
    questions = parse_csv(input_file)
    logger.info(f"Found {len(questions)} questions to test")
    
    # Limit to first 1 question for testing (1 batch)
    questions = questions[:10]
    logger.info(f"Testing first {len(questions)} question(s) in parallel")
    
    # Initialize test runner
    runner = TestRunner(API_URL)
    
    # Run tests in parallel with unique user IDs
    results = []
    
    def run_single_test(q_data: Dict, index: int) -> Dict:
        """Run a single test with unique user ID."""
        # Generate unique user ID for each parallel test
        unique_user_id = f"test_parallel_{q_data['category']}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[Parallel Test {index+1}/{len(questions)}] {q_data['category']}")
        logger.info(f"User ID: {unique_user_id}")
        logger.info(f"Question: {q_data['question']}")
        
        try:
            result = runner.run_multi_turn_test(
                question=q_data["question"],
                expected_answer=q_data["expected_answer"],
                category=q_data["category"],
                owner=q_data["owner"],
                user_id=unique_user_id  # Use unique user ID for context management
            )
            logger.info(f"[Test {index+1}] ✓ Completed - Score: {result['accuracy_score']:.1f}/10, Turns: {result['turns']}")
            return result
            
        except Exception as e:
            logger.error(f"[Test {index+1}] ✗ Error testing question: {e}")
            return {
                "category": q_data["category"],
                "question": q_data["question"],
                "expected_answer": q_data["expected_answer"],
                "final_answer": f"Test Error: {str(e)}",
                "accuracy_score": 0.0,
                "turns": 0,
                "conversation_turns": [],
                "sources_count": 0,
                "errors": [str(e)],
                "owner": q_data["owner"]
            }
    
    # Run tests in parallel using ThreadPoolExecutor
    logger.info(f"\n{'='*80}")
    logger.info("Starting parallel test execution...")
    logger.info(f"{'='*80}")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tests
        future_to_test = {
            executor.submit(run_single_test, q_data, i): (i, q_data) 
            for i, q_data in enumerate(questions)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_test):
            index, q_data = future_to_test[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"[Test {index+1}] Unexpected error: {e}")
                results.append({
                    "category": q_data["category"],
                    "question": q_data["question"],
                    "expected_answer": q_data["expected_answer"],
                    "final_answer": f"Unexpected Error: {str(e)}",
                    "accuracy_score": 0.0,
                    "turns": 0,
                    "conversation_turns": [],
                    "sources_count": 0,
                    "errors": [str(e)],
                    "owner": q_data["owner"]
                })
    
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"Parallel execution completed in {elapsed_time:.2f} seconds")
    logger.info(f"{'='*80}")
    
    # Sort results by original question order
    results = sorted(results, key=lambda x: next((i for i, q in enumerate(questions) if q["question"] == x["question"]), 0))
    
    # Save results
    runner.save_results(results, output_file)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*80}")
    total = len(results)
    avg_score = sum(r["accuracy_score"] for r in results) / total if total > 0 else 0
    avg_turns = sum(r["turns"] for r in results) / total if total > 0 else 0
    total_errors = sum(1 for r in results if r["errors"])
    
    logger.info(f"Total Questions: {total}")
    logger.info(f"Average Accuracy Score: {avg_score:.2f}/10")
    logger.info(f"Average Turns: {avg_turns:.2f}")
    logger.info(f"Questions with Errors: {total_errors}")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

