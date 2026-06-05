#!/usr/bin/env python3
"""
Test script to verify original question tracking and context preservation.
"""

import sys
sys.path.insert(0, '/home/user/conversational_rag')

from conversation_manager import ConversationManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_original_question_tracking():
    """Test that original questions are properly tracked."""

    # Create conversation manager
    conv_manager = ConversationManager()

    # Test user ID
    test_user = "test_context_user"

    # Clear any existing history
    conv_manager.clear_history(test_user)

    # Scenario: User asks a question, system asks for clarification, user answers
    logger.info("\n" + "="*80)
    logger.info("TEST: Original Question Tracking")
    logger.info("="*80)

    # Turn 1: User asks original question
    original_question = "What is the maternity leave policy?"
    conv_manager.add_message(
        test_user,
        "user",
        original_question,
        {"is_original_question": True}
    )
    logger.info(f"Turn 1 - User: {original_question}")

    # Turn 1: System asks for clarification
    clarification_q = "I need a bit more information to help you. Which country are you asking about?"
    conv_manager.add_message(test_user, "assistant", clarification_q)
    logger.info(f"Turn 1 - Assistant: {clarification_q}")

    # Turn 2: User answers clarification
    answer1 = "Lebanon"
    conv_manager.add_message(test_user, "user", answer1)
    logger.info(f"Turn 2 - User: {answer1}")

    # Turn 2: System asks another clarification
    clarification_q2 = "What is your position?"
    conv_manager.add_message(test_user, "assistant", clarification_q2)
    logger.info(f"Turn 2 - Assistant: {clarification_q2}")

    # Turn 3: User answers second clarification
    answer2 = "Manager"
    conv_manager.add_message(test_user, "user", answer2)
    logger.info(f"Turn 3 - User: {answer2}")

    # Now test: Can we retrieve the original question?
    logger.info("\n" + "-"*80)
    logger.info("Testing get_original_question()...")
    logger.info("-"*80)

    retrieved_original = conv_manager.get_original_question(test_user, within_last_n=10)

    logger.info(f"\nRetrieved Original Question: {retrieved_original}")
    logger.info(f"Expected: {original_question}")

    if retrieved_original == original_question:
        logger.info("✅ SUCCESS: Original question correctly retrieved!")
    else:
        logger.error(f"❌ FAIL: Expected '{original_question}', got '{retrieved_original}'")

    # Test conversation history
    logger.info("\n" + "-"*80)
    logger.info("Full Conversation History:")
    logger.info("-"*80)
    history = conv_manager.get_history(test_user)
    for i, msg in enumerate(history, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        metadata = msg.get("metadata", {})
        is_original = metadata.get("is_original_question", False)
        marker = " [ORIGINAL]" if is_original else ""
        logger.info(f"{i}. {role.upper()}: {content}{marker}")

    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)

    # Cleanup
    conv_manager.clear_history(test_user)

    return retrieved_original == original_question


if __name__ == "__main__":
    success = test_original_question_tracking()
    sys.exit(0 if success else 1)
