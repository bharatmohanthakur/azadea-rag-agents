"""
Best-Guess Answering Strategy - Answer first, clarify later.
Implements the ChatGPT/Claude/Gemini approach of providing immediate value.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger("BestGuessAnswering")


class AnswerConfidence(Enum):
    """Confidence levels for answers."""
    HIGH = "high"  # 0.7-1.0: Answer directly, no hedging needed
    MEDIUM = "medium"  # 0.4-0.7: Answer with hedging/assumptions
    LOW = "low"  # 0.2-0.4: Provide general answer + ONE clarification
    VERY_LOW = "very_low"  # 0-0.2: Must ask ONE specific question


class BestGuessAnswering:
    """
    Implements best-guess answering strategy.
    Philosophy: Better to give a slightly uncertain answer immediately
    than to interrogate the user with multiple questions.
    """

    def __init__(self, llm_client, deployment_name: str):
        """
        Initialize best-guess answering.

        Args:
            llm_client: LLM client
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name

        # Hedging templates for different confidence levels
        self.hedging_templates = {
            AnswerConfidence.HIGH: [
                "{answer}",
            ],
            AnswerConfidence.MEDIUM: [
                "Based on {assumption}, {answer}\n\nIf you meant something different, let me know!",
                "Typically, {answer}\n\nThis may vary depending on {variable} - let me know if you need specifics.",
                "{answer}\n\nNote: This assumes {assumption}. Different circumstances may have different rules.",
            ],
            AnswerConfidence.LOW: [
                "Here's the general answer:\n\n{answer}\n\nFor more specific information, could you clarify {question}?",
                "The typical case is:\n\n{answer}\n\nTo give you exact details for your situation, which {question}?",
            ]
        }

        # Common assumptions for HR queries
        self.common_assumptions = {
            "country": "Lebanon (headquarters)",
            "position": "staff-level position",
            "leave_type": "annual leave",
            "employment_type": "full-time permanent",
            "brand": "Azadea Group corporate"
        }

    def assess_confidence(
        self,
        query: str,
        retrieved_context: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[AnswerConfidence, List[str]]:
        """
        Assess confidence in answering query without additional clarification.

        Args:
            query: User query
            retrieved_context: Retrieved RAG context
            sources: Retrieved sources

        Returns:
            Tuple of (confidence level, list of ambiguous variables)
        """
        ambiguous_vars = []
        query_lower = query.lower()

        # Check for explicit specifications in query
        has_country = any(country in query_lower for country in
                         ["lebanon", "saudi", "uae", "egypt", "kuwait", "qatar", "jordan"])
        has_position = any(pos in query_lower for pos in
                          ["manager", "staff", "senior", "executive", "director", "employee"])
        has_brand = any(brand in query_lower for brand in
                       ["azadea", "oysho", "zara", "mango", "pull&bear"])

        # Check what's missing
        needs_country = self._query_needs_country(query_lower, retrieved_context)
        needs_position = self._query_needs_position(query_lower, retrieved_context)

        if needs_country and not has_country:
            ambiguous_vars.append("country/location")
        if needs_position and not has_position:
            ambiguous_vars.append("position/role")

        # Check if context provides enough information
        context_richness = len(retrieved_context) / 1000  # Rough metric
        source_count = len(sources)

        # Determine confidence
        if not ambiguous_vars and context_richness > 2 and source_count >= 3:
            confidence = AnswerConfidence.HIGH
        elif len(ambiguous_vars) <= 1 and source_count >= 2:
            confidence = AnswerConfidence.MEDIUM
        elif len(ambiguous_vars) <= 2:
            confidence = AnswerConfidence.LOW
        else:
            confidence = AnswerConfidence.VERY_LOW

        logger.info(f"Confidence: {confidence.value}, Ambiguous vars: {ambiguous_vars}")
        return confidence, ambiguous_vars

    def _query_needs_country(self, query: str, context: str) -> bool:
        """Check if query answer varies by country."""
        country_dependent_keywords = [
            "leave", "vacation", "holiday", "insurance", "benefit",
            "salary", "compensation", "allowance", "policy"
        ]
        return any(keyword in query for keyword in country_dependent_keywords)

    def _query_needs_position(self, query: str, context: str) -> bool:
        """Check if query answer varies by position."""
        position_dependent_keywords = [
            "leave", "vacation", "benefit", "insurance",
            "allowance", "eligibility", "entitled"
        ]
        return any(keyword in query for keyword in position_dependent_keywords)

    def make_assumptions(
        self,
        ambiguous_vars: List[str],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Make intelligent assumptions for ambiguous variables.

        Args:
            ambiguous_vars: List of ambiguous variables
            user_profile: Optional user profile with extracted info

        Returns:
            Dictionary of assumptions made
        """
        assumptions = {}

        for var in ambiguous_vars:
            # Check user profile first
            if user_profile and var in user_profile:
                assumptions[var] = user_profile[var]
            # Fall back to common assumptions
            elif var in self.common_assumptions:
                assumptions[var] = self.common_assumptions[var]
            else:
                assumptions[var] = "typical/common case"

        return assumptions

    def generate_hedged_answer(
        self,
        base_answer: str,
        confidence: AnswerConfidence,
        assumptions: Dict[str, str],
        ambiguous_vars: List[str]
    ) -> str:
        """
        Generate answer with appropriate hedging based on confidence.

        Args:
            base_answer: The base answer text
            confidence: Confidence level
            assumptions: Assumptions made
            ambiguous_vars: Ambiguous variables

        Returns:
            Hedged answer text
        """
        import random

        if confidence == AnswerConfidence.HIGH:
            # No hedging needed
            return base_answer

        # Select appropriate template
        templates = self.hedging_templates.get(confidence, [])
        template = random.choice(templates) if templates else "{answer}"

        # Format assumptions
        if assumptions:
            assumption_text = ", ".join([f"{k}: {v}" for k, v in assumptions.items()])
        else:
            assumption_text = "typical circumstances"

        # Format variable question
        variable_text = " or ".join(ambiguous_vars) if ambiguous_vars else "your specific situation"

        # Apply template
        hedged = template.format(
            answer=base_answer,
            assumption=assumption_text,
            variable=variable_text,
            question=variable_text
        )

        return hedged

    async def answer_with_best_guess(
        self,
        query: str,
        retrieved_context: str,
        sources: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using best-guess strategy.

        Args:
            query: User query
            retrieved_context: Retrieved context
            sources: Retrieved sources
            user_profile: Optional user profile
            system_prompt: Optional system prompt

        Returns:
            Dictionary with answer, confidence, assumptions
        """
        # Assess confidence
        confidence, ambiguous_vars = self.assess_confidence(query, retrieved_context, sources)

        # Make assumptions
        assumptions = self.make_assumptions(ambiguous_vars, user_profile)

        # Build enhanced context with assumptions
        if assumptions:
            assumption_text = "\n".join([f"Assuming {k}: {v}" for k, v in assumptions.items()])
            enhanced_context = f"{retrieved_context}\n\n**Assumptions Made**:\n{assumption_text}"
        else:
            enhanced_context = retrieved_context

        # Generate base answer using LLM
        if not system_prompt:
            system_prompt = """You are a helpful HR assistant. Use step-by-step reasoning to provide best-guess answers.

**STEP 1 - CHAIN OF THOUGHT ANALYSIS:**
Think through these steps:

1. Context evaluation:
   - What information is explicitly available in the context?
   - What information might be ambiguous or missing?
   - What is the user really asking for?

2. Ambiguity assessment:
   - Is Country/Location specified? If not, what's the most common case?
   - Is Job Position specified? If not, what's the typical scenario?
   - Are there other variables that might affect the answer?

3. Best-guess strategy:
   - For missing Country → assume headquarters/most common location
   - For missing Position → assume staff-level unless context suggests otherwise
   - For missing details → use the most typical/general case

4. Assumption clarity:
   - What assumptions am I making?
   - Should I state these assumptions in my answer?
   - Can I provide a useful answer even with these assumptions?

**STEP 2 - ANSWER GENERATION:**
IMPORTANT: Even if some information is ambiguous, provide a useful answer based on the most common/typical case.

CRITICAL:
- Do NOT say "I need more information"
- DO state your assumptions clearly
- DO provide the best answer you can with available information
- DO make it clear what you're assuming so the user can correct if needed"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{enhanced_context}\n\nQuestion: {query}"}
        ]

        try:
            response = self.llm_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.1,
                max_tokens=10000
            )

            base_answer = response.choices[0].message.content

            # Apply hedging
            final_answer = self.generate_hedged_answer(
                base_answer=base_answer,
                confidence=confidence,
                assumptions=assumptions,
                ambiguous_vars=ambiguous_vars
            )

            return {
                "answer": final_answer,
                "confidence": confidence.value,
                "assumptions": assumptions,
                "ambiguous_vars": ambiguous_vars,
                "sources": sources,
                "needs_clarification": confidence == AnswerConfidence.VERY_LOW
            }

        except Exception as e:
            logger.error(f"Error generating best-guess answer: {e}")
            return {
                "answer": "I apologize, but I encountered an error generating the answer.",
                "confidence": "error",
                "assumptions": {},
                "ambiguous_vars": [],
                "sources": sources,
                "needs_clarification": False
            }


# Global instance
_best_guess_answering: Optional[BestGuessAnswering] = None


def get_best_guess_answering() -> Optional[BestGuessAnswering]:
    """Get global best-guess answering instance."""
    return _best_guess_answering


def init_best_guess_answering(llm_client, deployment_name: str):
    """
    Initialize global best-guess answering.

    Args:
        llm_client: LLM client
        deployment_name: Azure deployment name
    """
    global _best_guess_answering
    _best_guess_answering = BestGuessAnswering(
        llm_client=llm_client,
        deployment_name=deployment_name
    )
    logger.info("Initialized best-guess answering strategy")
