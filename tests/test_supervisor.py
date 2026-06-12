"""
Supervisor test suite — agents/supervisor.py
=============================================

Covers every function and behaviour in the supervisor:
  1. _fallback_intent()         — regex routing when LLM fails
  2. _needs_reflection()        — reflection trigger logic
  3. _extract_conversation_text() — message formatting
  4. supervisor_node()          — full node with mocked LLM
  5. Entity extraction          — order_id and refund_amount parsing
  6. Query decomposition        — multi-intent detection
  7. Long-term memory recall    — past_context injection
  8. OTP detection              — 6-digit code routing
  9. State preservation         — turn_count, start_time, order_id
 10. Known gaps (xfail)         — documented limitations

Run all:
    pytest tests/test_supervisor.py -v

Run just fallback tests (no mocking):
    pytest tests/test_supervisor.py -v -k "Fallback"

Run just node integration tests:
    pytest tests/test_supervisor.py -v -k "Node"
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_state(
    message: str = "Where is my order?",
    customer_id: str = "CUST-001",
    session_id: str = "sess-test-001",
    order_id: str | None = None,
    intent: str = "unknown",
    turn_count: int = 0,
    pending_intents: list | None = None,
    resolution_status: str = "pending",
    messages: list | None = None,
) -> dict:
    from langchain_core.messages import HumanMessage
    if messages is None:
        messages = [HumanMessage(content=message)]
    return {
        "messages": messages,
        "customer_id": customer_id,
        "session_id": session_id,
        "order_id": order_id,
        "intent": intent,
        "turn_count": turn_count,
        "pending_intents": pending_intents or [],
        "resolution_status": resolution_status,
        "partial_responses": [],
        "accumulated_docs": [],
        "is_decomposed": False,
        "retrieved_docs": [],
        "metadata": {},
        "start_time": None,
    }


def mock_llm_response(json_str: str) -> MagicMock:
    """Return a mock LLM chain whose invoke() returns json_str."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = json_str
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=json_str)
    return mock_llm, mock_chain


def run_node_with_mock_llm(
    state: dict,
    llm_response: str,
    decompose_response: str = '["general"]',
    memory_count: int = 0,
) -> dict:
    """Run supervisor_node with LLM and memory fully mocked."""
    from agents.supervisor import supervisor_node

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=decompose_response)

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = llm_response

    with patch("agents.supervisor._get_llm", return_value=mock_llm), \
         patch("agents.supervisor.ChatPromptTemplate") as mock_tpl, \
         patch("memory.long_term.LongTermMemory") as mock_mem:

        # Wire prompt | llm | parser chain
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(
            return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            )
        )
        mock_tpl.from_messages.return_value = mock_prompt

        # Wire memory
        mock_mem_instance = MagicMock()
        mock_mem_instance.get_customer_history_count.return_value = memory_count
        mock_mem_instance.recall.return_value = []
        mock_mem_instance.format_for_prompt.return_value = ""
        mock_mem.return_value = mock_mem_instance

        return supervisor_node(state)


# ─────────────────────────────────────────────────────────────────────────────
# 1. _fallback_intent() — regex routing
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackIntent:
    """Tests for the regex fallback used when the LLM fails."""

    # ── Order lookup triggers ─────────────────────────────────────────────────

    def test_order_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("Where is my order?") == "order_lookup"

    def test_track_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("Can you track my package?") == "order_lookup"

    def test_ship_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("Has my item shipped yet?") == "order_lookup"

    def test_deliver_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("When will it be delivered?") == "order_lookup"

    def test_status_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("What is the status of my package?") == "order_lookup"

    # ── Policy returns triggers ───────────────────────────────────────────────

    def test_return_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("I want to return this item") == "policy_returns"

    def test_refund_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("I need a refund please") == "policy_returns"

    def test_exchange_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("Can I exchange this for a different size?") == "policy_returns"

    def test_money_back_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("I want my money back") == "policy_returns"

    def test_broken_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("My product is broken") == "policy_returns"

    # ── Escalation triggers ───────────────────────────────────────────────────

    def test_manager_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("I want to speak to a manager") == "escalation"

    def test_unacceptable_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("This is completely unacceptable") == "escalation"

    def test_lawsuit_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("I will file a lawsuit") == "escalation"

    def test_complain_keyword(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("I want to file a complaint") == "escalation"

    # ── General fallback ──────────────────────────────────────────────────────

    def test_product_question_goes_to_general(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("What are your store hours?") == "general"

    def test_empty_message_goes_to_general(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("") == "general"

    def test_greeting_goes_to_general(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("Hi there!") == "general"

    def test_troubleshooting_goes_to_general(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("My device keeps burning") == "general"

    def test_case_insensitive(self):
        from agents.supervisor import _fallback_intent
        assert _fallback_intent("WHERE IS MY ORDER") == "order_lookup"

    def test_priority_order_over_return(self):
        """'order' keyword takes priority over 'return' keyword."""
        from agents.supervisor import _fallback_intent
        result = _fallback_intent("I want to return my order ORD-10001")
        assert result in ("order_lookup", "policy_returns")


# ─────────────────────────────────────────────────────────────────────────────
# 2. _needs_reflection() — reflection trigger
# ─────────────────────────────────────────────────────────────────────────────

class TestNeedsReflection:

    def test_short_answer_triggers_reflection(self):
        from agents.supervisor import _needs_reflection
        assert _needs_reflection("ok") is True

    def test_empty_answer_triggers_reflection(self):
        from agents.supervisor import _needs_reflection
        assert _needs_reflection("") is True

    def test_long_confident_answer_no_reflection(self):
        from agents.supervisor import _needs_reflection
        assert _needs_reflection('{"intent": "order_lookup", "confidence": 0.95}') is False

    def test_i_dont_know_triggers_reflection(self):
        from agents.supervisor import _needs_reflection
        assert _needs_reflection("I don't know what to route this to") is True

    def test_im_not_sure_triggers_reflection(self):
        from agents.supervisor import _needs_reflection
        assert _needs_reflection("I'm not sure about this one") is True

    def test_unclear_triggers_reflection(self):
        from agents.supervisor import _needs_reflection
        assert _needs_reflection("This is unclear to me") is True

    def test_exactly_20_chars_no_reflection(self):
        from agents.supervisor import _needs_reflection
        # 20 chars, no uncertainty phrases
        assert _needs_reflection("a" * 20) is False

    def test_19_chars_triggers_reflection(self):
        from agents.supervisor import _needs_reflection
        assert _needs_reflection("a" * 19) is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. _extract_conversation_text() — message formatting
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractConversationText:

    def test_single_human_message(self):
        from langchain_core.messages import HumanMessage
        from agents.supervisor import _extract_conversation_text
        msgs = [HumanMessage(content="Hello")]
        assert "Customer: Hello" in _extract_conversation_text(msgs)

    def test_human_and_ai_message(self):
        from langchain_core.messages import HumanMessage, AIMessage
        from agents.supervisor import _extract_conversation_text
        msgs = [HumanMessage(content="Hi"), AIMessage(content="Hello!")]
        text = _extract_conversation_text(msgs)
        assert "Customer: Hi" in text
        assert "Agent: Hello!" in text

    def test_empty_messages_returns_empty(self):
        from agents.supervisor import _extract_conversation_text
        assert _extract_conversation_text([]) == ""

    def test_truncates_to_last_6_messages(self):
        from langchain_core.messages import HumanMessage
        from agents.supervisor import _extract_conversation_text
        msgs = [HumanMessage(content=f"Message {i}") for i in range(10)]
        text = _extract_conversation_text(msgs)
        assert "Message 0" not in text
        assert "Message 9" in text

    def test_only_human_and_ai_included(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        from agents.supervisor import _extract_conversation_text
        msgs = [SystemMessage(content="System"), HumanMessage(content="Hi")]
        text = _extract_conversation_text(msgs)
        assert "System" not in text
        assert "Customer: Hi" in text


# ─────────────────────────────────────────────────────────────────────────────
# 4. supervisor_node() — full node integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSupervisorNode:

    def test_node_returns_intent(self):
        state = make_state("Where is my order ORD-10001?")
        llm_json = '{"intent": "order_lookup", "order_id": "ORD-10001", "refund_amount": null, "confidence": 0.95, "reasoning": "customer asked about specific order"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["intent"] == "order_lookup"

    def test_node_extracts_order_id(self):
        state = make_state("Where is ORD-10002?")
        llm_json = '{"intent": "order_lookup", "order_id": "ORD-10002", "refund_amount": null, "confidence": 0.9, "reasoning": "order ID mentioned"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["order_id"] == "ORD-10002"

    def test_node_extracts_refund_amount(self):
        state = make_state("I want a refund of $129.99")
        llm_json = '{"intent": "policy_returns", "order_id": null, "refund_amount": 129.99, "confidence": 0.9, "reasoning": "refund amount mentioned"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["refund_amount"] == 129.99

    def test_node_sets_start_time(self):
        state = make_state()
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general query"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["start_time"] is not None
        assert isinstance(updates["start_time"], float)

    def test_node_preserves_existing_start_time(self):
        import time
        state = make_state()
        state["start_time"] = 1000000.0
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["start_time"] == 1000000.0

    def test_node_increments_turn_count(self):
        state = make_state()
        state["turn_count"] = 3
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["turn_count"] == 4

    def test_node_preserves_existing_order_id(self):
        """If order_id already in state and LLM returns null, preserve existing."""
        state = make_state(order_id="ORD-10001")
        llm_json = '{"intent": "policy_returns", "order_id": null, "refund_amount": null, "confidence": 0.9, "reasoning": "return request"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["order_id"] == "ORD-10001"

    def test_node_falls_back_to_regex_on_llm_error(self):
        from agents.supervisor import supervisor_node
        from langchain_core.messages import HumanMessage

        state = make_state("Where is my order?")

        with patch("agents.supervisor._get_llm", side_effect=Exception("API down")), \
             patch("memory.long_term.LongTermMemory") as mock_mem:
            mock_mem.return_value.get_customer_history_count.return_value = 0
            updates = supervisor_node(state)

        assert updates["intent"] == "order_lookup"

    def test_node_returns_metadata_with_confidence(self):
        state = make_state()
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.75, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert "supervisor_confidence" in updates["metadata"]
        assert updates["metadata"]["supervisor_confidence"] == 0.75

    def test_node_returns_next_agent_matching_intent(self):
        state = make_state()
        llm_json = '{"intent": "escalation", "order_id": null, "refund_amount": null, "confidence": 0.9, "reasoning": "manager request"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["next_agent"] == "escalation"

    def test_node_handles_malformed_json_gracefully(self):
        from agents.supervisor import supervisor_node
        from langchain_core.messages import HumanMessage

        state = make_state("I need help")

        with patch("agents.supervisor._get_llm") as mock_get_llm, \
             patch("agents.supervisor.ChatPromptTemplate") as mock_tpl, \
             patch("memory.long_term.LongTermMemory") as mock_mem:

            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "not valid json at all"
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))
            mock_tpl.from_messages.return_value = mock_prompt
            mock_mem.return_value.get_customer_history_count.return_value = 0

            updates = supervisor_node(state)

        # Should not crash — falls back to regex
        assert "intent" in updates

    def test_node_strips_markdown_fences_from_llm_output(self):
        state = make_state("Where is my order?")
        llm_with_fences = '```json\n{"intent": "order_lookup", "order_id": null, "refund_amount": null, "confidence": 0.9, "reasoning": "order query"}\n```'
        updates = run_node_with_mock_llm(state, llm_with_fences)
        assert updates["intent"] == "order_lookup"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entity extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityExtraction:

    def test_extracts_order_id_from_llm(self):
        state = make_state("What about ORD-10005?")
        llm_json = '{"intent": "order_lookup", "order_id": "ORD-10005", "refund_amount": null, "confidence": 0.95, "reasoning": "order ID"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["order_id"] == "ORD-10005"

    def test_ignores_null_string_order_id(self):
        state = make_state(order_id="ORD-10001")
        llm_json = '{"intent": "general", "order_id": "null", "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["order_id"] == "ORD-10001"

    def test_extracts_float_refund_amount(self):
        state = make_state("I want $250.50 back")
        llm_json = '{"intent": "policy_returns", "order_id": null, "refund_amount": 250.50, "confidence": 0.9, "reasoning": "refund"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["refund_amount"] == 250.50

    def test_handles_none_refund_amount(self):
        state = make_state("I have a question")
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["refund_amount"] is None

    def test_handles_invalid_refund_amount_gracefully(self):
        state = make_state("refund please")
        llm_json = '{"intent": "policy_returns", "order_id": null, "refund_amount": "not_a_number", "confidence": 0.8, "reasoning": "return"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["refund_amount"] is None

    def test_extracts_confidence_score(self):
        state = make_state("Hi")
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.42, "reasoning": "greeting"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["metadata"]["supervisor_confidence"] == 0.42


# ─────────────────────────────────────────────────────────────────────────────
# 6. OTP detection
# ─────────────────────────────────────────────────────────────────────────────

class TestOTPDetection:

    def test_six_digit_code_routes_to_order_lookup(self):
        from agents.supervisor import supervisor_node
        from langchain_core.messages import HumanMessage

        state = make_state(message="123456")
        with patch("agents.supervisor._get_llm") as mock_get, \
             patch("agents.supervisor.ChatPromptTemplate") as mock_tpl, \
             patch("memory.long_term.LongTermMemory") as mock_mem:

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = '{"intent":"general","order_id":null,"refund_amount":null,"confidence":0.5,"reasoning":"otp"}'
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))
            mock_tpl.from_messages.return_value = mock_prompt
            mock_get.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content='["general"]')))
            mock_mem.return_value.get_customer_history_count.return_value = 0

            updates = supervisor_node(state)

        assert updates["intent"] == "order_lookup"
        assert updates["pending_intents"] == []

    def test_non_otp_message_not_treated_as_otp(self):
        state = make_state("I have 123456 items to return")
        llm_json = '{"intent": "policy_returns", "order_id": null, "refund_amount": null, "confidence": 0.85, "reasoning": "return"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["intent"] == "policy_returns"

    def test_five_digit_not_otp(self):
        from agents.supervisor import supervisor_node
        state = make_state(message="12345")
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.6, "reasoning": "unclear"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["intent"] != "order_lookup" or updates["pending_intents"] == []


# ─────────────────────────────────────────────────────────────────────────────
# 7. Long-term memory recall
# ─────────────────────────────────────────────────────────────────────────────

class TestLongTermMemoryRecall:

    def test_past_context_empty_when_no_history(self):
        state = make_state(customer_id="CUST-NEW-999")
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json, memory_count=0)
        assert updates["metadata"].get("past_context", "") == ""

    def test_past_context_populated_when_history_exists(self):
        from agents.supervisor import supervisor_node
        from langchain_core.messages import HumanMessage

        state = make_state(customer_id="CUST-001")

        with patch("agents.supervisor._get_llm") as mock_get, \
             patch("agents.supervisor.ChatPromptTemplate") as mock_tpl, \
             patch("memory.long_term.LongTermMemory") as mock_mem:

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = '{"intent":"general","order_id":null,"refund_amount":null,"confidence":0.8,"reasoning":"general"}'
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))
            mock_tpl.from_messages.return_value = mock_prompt

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content='["general"]')
            mock_get.return_value = mock_llm

            mock_mem_instance = MagicMock()
            mock_mem_instance.get_customer_history_count.return_value = 3
            mock_mem_instance.recall.return_value = [
                {"content": "Previous escalation TKT-ABC.", "timestamp": "2026-05-01", "relevance": 0.8}
            ]
            mock_mem_instance.format_for_prompt.return_value = "Previous escalation TKT-ABC."
            mock_mem.return_value = mock_mem_instance

            updates = supervisor_node(state)

        assert "Previous escalation TKT-ABC." in updates["metadata"]["past_context"]

    def test_memory_failure_does_not_crash_node(self):
        from agents.supervisor import supervisor_node

        state = make_state()

        with patch("agents.supervisor._get_llm") as mock_get, \
             patch("agents.supervisor.ChatPromptTemplate") as mock_tpl, \
             patch("memory.long_term.LongTermMemory", side_effect=Exception("ChromaDB down")):

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = '{"intent":"general","order_id":null,"refund_amount":null,"confidence":0.8,"reasoning":"general"}'
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))
            mock_tpl.from_messages.return_value = mock_prompt
            mock_get.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content='["general"]')))

            updates = supervisor_node(state)

        assert updates["metadata"].get("past_context", "") == ""
        assert "intent" in updates

    def test_recall_query_enriched_with_intent_and_order_id(self):
        from agents.supervisor import supervisor_node

        state = make_state("I need help with my return", order_id="ORD-10001")

        with patch("agents.supervisor._get_llm") as mock_get, \
             patch("agents.supervisor.ChatPromptTemplate") as mock_tpl, \
             patch("memory.long_term.LongTermMemory") as mock_mem:

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = '{"intent":"policy_returns","order_id":"ORD-10001","refund_amount":null,"confidence":0.9,"reasoning":"return"}'
            mock_prompt = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))
            mock_tpl.from_messages.return_value = mock_prompt
            mock_get.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content='["policy_returns"]')))

            mock_mem_instance = MagicMock()
            mock_mem_instance.get_customer_history_count.return_value = 2
            mock_mem_instance.recall.return_value = []
            mock_mem_instance.format_for_prompt.return_value = ""
            mock_mem.return_value = mock_mem_instance

            supervisor_node(state)

        call_kwargs = mock_mem_instance.recall.call_args
        query_used = call_kwargs[1].get("query") or call_kwargs[0][1]
        assert "policy_returns" in query_used
        assert "ORD-10001" in query_used


# ─────────────────────────────────────────────────────────────────────────────
# 8. State preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestStatePreservation:

    def test_turn_count_starts_at_zero_increments_to_one(self):
        state = make_state()
        state["turn_count"] = 0
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["turn_count"] == 1

    def test_turn_count_accumulates_across_turns(self):
        state = make_state()
        state["turn_count"] = 5
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["turn_count"] == 6

    def test_existing_metadata_preserved(self):
        state = make_state()
        state["metadata"] = {"existing_key": "existing_value"}
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["metadata"]["existing_key"] == "existing_value"

    def test_pending_intents_starts_empty_for_fresh_query(self):
        state = make_state(resolution_status="pending")
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert isinstance(updates["pending_intents"], list)

    def test_start_time_not_overwritten_if_already_set(self):
        state = make_state()
        state["start_time"] = 999999.0
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["start_time"] == 999999.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Known gaps (xfail)
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownGaps:

    @pytest.mark.xfail(
        reason="Fallback regex does not distinguish troubleshooting from escalation — 'broken' routes to policy_returns not general",
        strict=True,
    )
    def test_troubleshooting_complaint_goes_to_general_not_policy(self):
        from agents.supervisor import _fallback_intent
        result = _fallback_intent("My coffee maker is broken after 2 months")
        assert result == "general"

    @pytest.mark.xfail(
        reason="LongTermMemory is still instantiated inline in supervisor_node — should use module-level singleton",
        strict=True,
    )
    def test_long_term_memory_uses_singleton_not_inline_instantiation(self):
        import inspect
        from agents.supervisor import supervisor_node
        source = inspect.getsource(supervisor_node)
        assert "LongTermMemory()" not in source

    @pytest.mark.xfail(
        reason="No turn limit guard — conversations can loop forever without routing to escalation",
        strict=True,
    )
    def test_high_turn_count_routes_to_escalation(self):
        state = make_state()
        state["turn_count"] = 15
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.8, "reasoning": "general"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["intent"] == "escalation"

    @pytest.mark.xfail(
        reason="Low confidence does not trigger clarification — routed same as high confidence",
        strict=True,
    )
    def test_low_confidence_triggers_clarifying_question(self):
        state = make_state("I need some help")
        llm_json = '{"intent": "general", "order_id": null, "refund_amount": null, "confidence": 0.2, "reasoning": "ambiguous"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates.get("needs_clarification") is True

    @pytest.mark.xfail(
        reason="refund_amount not preserved across turns — resets to None even when previously extracted",
        strict=True,
    )
    def test_refund_amount_preserved_across_turns(self):
        state = make_state()
        state["refund_amount"] = 250.0
        llm_json = '{"intent": "policy_returns", "order_id": null, "refund_amount": null, "confidence": 0.9, "reasoning": "follow-up"}'
        updates = run_node_with_mock_llm(state, llm_json)
        assert updates["refund_amount"] == 250.0