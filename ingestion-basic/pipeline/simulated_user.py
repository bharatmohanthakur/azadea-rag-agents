
import os
import random
import json
import asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration (Reuse existing env vars)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o")

client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

class UserGoal(BaseModel):
    persona: str = Field(description="Who the user is (e.g. 'Manager in Bahrain', 'New Joiner')")
    goal_description: str = Field(description="What information they need locally")
    initial_query: str = Field(description="The first message to send to the bot")
    expected_answer_facts: List[str] = Field(description="Key facts from the doc that must be in the final answer")

class TurnEvaluation(BaseModel):
    is_session_complete: bool = Field(description="True if the user's goal is met OR if the bot failed irretrievably")
    user_response: Optional[str] = Field(default="", description="The next message to send (if not complete). Empty if complete.")
    grade: int = Field(description="Score 0-10. 10=Perfect answer, 0=Complete failure. Only meaningful if session is complete.")
    reasoning: str = Field(description="Why this grade was given.")

class SimulatedUser:
    def __init__(self, doc_content: str, source_filename: str):
        self.doc_content = doc_content
        self.source_filename = source_filename
        self.history = [] # List of {"role": "user/assistant", "content": "..."}
        self.goal: Optional[UserGoal] = None

    async def generate_goal(self, scenario_type: str = "balanced"):
        """Reads the doc and invents a user persona and goal based on scenario_type."""
        
        scenario_instruction = ""
        if scenario_type == "image_workflow":
            scenario_instruction = "CRITICAL: The user MUST ask about a visual diagram, flowchart, or image content described in the text. The goal is to test if the bot can 'see' the workflow."
        elif scenario_type == "dynamic_followup":
            scenario_instruction = "CRITICAL: Start with a VERY BROAD, GENERIC query (e.g. 'benefits', 'policy', 'allowance') that forces the bot to ask for clarification. The goal is to test multi-turn negotiation."
        else:
            scenario_instruction = "Create a balanced, realistic query. Mix of specific and slightly ambiguous."

        prompt = f"""You are a Creative User Simulator.
Read the following document snippet (Source: {self.source_filename}) and invent a realistic user scenario.

Document Content:
{self.doc_content[:15000]}... (truncated)

Task:
1. Create a persona relevant to this doc.
2. Define a specific information goal.
3. Formulate an INITIAL QUERY according to this SCENARIO: {scenario_instruction}
4. Extract the GROUND TRUTH facts from the doc that answer this specific goal.

Respond in JSON format only. Use this EXACT structure:
{{
  "persona": "string (e.g. 'Manager')",
  "goal_description": "string (e.g. 'I want to know...')",
  "initial_query": "string (e.g. 'What is...')",
  "expected_answer_facts": ["string", "string"]
}}
"""
        response = await client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        try:
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            data = json.loads(content)
            self.goal = UserGoal(**data)
            self.history.append({"role": "user", "content": self.goal.initial_query})
            return self.goal
        except Exception as e:
            print(f"Error generating goal: {e}")
            self.goal = UserGoal(
                persona="Generic User",
                goal_description="Understand source",
                initial_query=f"Summarize {self.source_filename}",
                expected_answer_facts=[]
            )
            return self.goal

    async def process_turn(self, assistant_reply: str) -> TurnEvaluation:
        """Decides what to do next based on assistant's reply."""
        self.history.append({"role": "assistant", "content": assistant_reply})
        
        prompt = f"""You are a Simulated User ({self.goal.persona}).
You are interacting with an AI Support Bot.
ROLE: You are the USER. Do NOT act as the Assistant. Do NOT give advice.
Your Goal: {self.goal.goal_description}
Expected Facts: {json.dumps(self.goal.expected_answer_facts)}

Conversation History:
{json.dumps(self.history[-4:], indent=2)}

Latest Assistant Reply: "{assistant_reply}"

Task:
Determine your next move.
1. **Is the goal met?** If the assistant provided the Expected Facts, the session is COMPLETE. Grade 10.
2. **Did the assistant ask a clarifying question?** (e.g. "Which country?", "Which role?").
   - If YES, checks the Document Content (below) to find the answer.
   - GENERATE a response answering the question (e.g. "I am in Lebanon").
   - Session is NOT complete.
3. **Did the assistant fail?** (e.g. "I don't know", or gave wrong info).
   - Session is COMPLETE. Grade 0-5.

Document Content (Reference):
{self.doc_content[:15000]}

Output JSON: {{ "is_session_complete": bool, "user_response": "...", "grade": int, "reasoning": "..." }}

Respond in JSON format only. Use this EXACT structure:
{{
  "is_session_complete": false,
  "user_response": "string (next message)",
  "grade": 0,
  "reasoning": "string"
}}
"""
        response = await client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        try:
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            data = json.loads(content)
            
            evaluation = TurnEvaluation(**data)
            
            if not evaluation.is_session_complete:
                self.history.append({"role": "user", "content": evaluation.user_response})
            
            return evaluation
        except Exception as e:
            print(f"Error in process_turn: {e}")
            return TurnEvaluation(is_session_complete=True, user_response="", grade=0, reasoning="Error in simulation logic")

if __name__ == "__main__":
    print("SimulatedUser class defined.")
