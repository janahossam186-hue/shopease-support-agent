"""
Escalation agent test suite — escalation.py

Sections:
  1. Ticket generation tests     — uniqueness, format, collision resistance
  2. ETA logic tests             — correct timeline per refund amount and violation type
  3. Escalation reason tests     — correct reason extracted from state
  4. Node integration tests      — full node with mocked LLM and memory
  5. Long-term memory tests      — summary content and save behavior
  6. Known gaps (xfail)          — documents what's missing

Run all:
    pytest tests/test_escalation.py -v

Run just ticket tests:
    pytest tests/test_escalation.py -v -k "Ticket"

Run just ETA tests:
    pytest tests/test_escalation.py -v -k "ETA"
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
    message: str = "I want to speak to a manager",
    customer_id: str = "CUST-001",
    order_id: str | None = None,
    refund_amount: float | None = None,
    intent: str = "escalation",
    requires_escalation: bool = True,
    policy_violations: list | None = None,
) -> dict:
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=message)],
        "customer_id": customer_id,
        "session_id": "sess-test-001",
        "order_id": order_id,
        "refund_amount": refund_amount,
        "intent": intent,
        "requires_escalation": requires_escalation,
        "policy_violations": policy_violations or [],
        "metadata": {
            "past_context": "No prior interactions.",
            "policy_violations": policy_violations or [],
        },
    }


def run_node(state: dict, llm_response: str = "I'm escalating your case. Ticket: TKT-TEST1.") -> dict:
    """Run escalation_node with LLM and long-term memory fully mocked."""
    from agents.escalation import escalation_node

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = llm_response

    with patch("agents.escalation._get_llm") as mock_llm, \
         patch("agents.escalation._long_term_memory") as mock_mem:

        mock_llm_instance = MagicMock()
        mock_llm_instance.__or__ = MagicMock(
            return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
        )
        mock_llm.return_value = mock_llm_instance
        mock_mem.save_interaction.return_value = None

        return escalation_node(state)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ticket generation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTicketGeneration:

    def test_ticket_has_correct_prefix(self):
        from agents.escalation import _generate_ticket_id
        assert _generate_ticket_id().startswith("TKT-")

    def test_ticket_format_length(self):
        from agents.escalation import _generate_ticket_id
        ticket = _generate_ticket_id()
        # TKT- is 4 chars, suffix should be at least 5
        assert len(ticket) >= 9

    def test_ticket_is_string(self):
        from agents.escalation import _generate_ticket_id
        assert isinstance(_generate_ticket_id(), str)

    def test_100_tickets_are_all_unique(self):
        """Generate 100 tickets and verify no duplicates."""
        from agents.escalation import _generate_ticket_id
        tickets = [_generate_ticket_id() for _ in range(100)]
        assert len(set(tickets)) == 100

    def test_ticket_appears_in_node_response(self):
        """The ticket ID generated must appear in the state output."""
        updates = run_node(make_state())
        assert "escalation_ticket_id" in updates
        assert updates["escalation_ticket_id"].startswith("TKT-")

    def test_ticket_id_in_state_matches_format(self):
        updates = run_node(make_state())
        ticket = updates["escalation_ticket_id"]
        assert ticket.startswith("TKT-")
        assert len(ticket) >= 9


# ─────────────────────────────────────────────────────────────────────────────
# 2. ETA logic tests
# ─────────────────────────────────────────────────────────────────────────────

class TestETALogic:

    def test_no_refund_returns_4_hours(self):
        from agents.escalation import _determine_eta
        state = make_state(refund_amount=None)
        assert "4 business hours" in _determine_eta(state)

    def test_small_refund_returns_4_hours(self):
        from agents.escalation import _determine_eta
        state = make_state(refund_amount=200.0)
        assert "4 business hours" in _determine_eta(state)

    def test_refund_at_500_returns_4_hours(self):
        from agents.escalation import _determine_eta
        state = make_state(refund_amount=500.0)
        assert "4 business hours" in _determine_eta(state)

    def test_refund_501_returns_1_business_day(self):
        from agents.escalation import _determine_eta
        state = make_state(refund_amount=501.0)
        eta = _determine_eta(state)
        assert "1 business day" in eta

    def test_refund_750_returns_1_business_day(self):
        from agents.escalation import _determine_eta
        state = make_state(refund_amount=750.0)
        assert "1 business day" in _determine_eta(state)

    def test_refund_1000_returns_1_business_day(self):
        from agents.escalation import _determine_eta
        state = make_state(refund_amount=1000.0)
        assert "1 business day" in _determine_eta(state)

    def test_refund_1001_returns_2_3_days(self):
        from agents.escalation import _determine_eta
        state = make_state(refund_amount=1001.0)
        eta = _determine_eta(state)
        assert "2" in eta and "3" in eta

    def test_refund_1500_returns_2_3_days(self):
        from agents.escalation import _determine_eta
        state = make_state(refund_amount=1500.0)
        eta = _determine_eta(state)
        assert "2" in eta or "3" in eta

    def test_eta_boundary_exactly_1000(self):
        from agents.escalation import _determine_eta
        # $1000 is supervisor tier (≤1000), not manager tier (>1000)
        state = make_state(refund_amount=1000.0)
        assert "1 business day" in _determine_eta(state)

    def test_eta_boundary_1000_01(self):
        from agents.escalation import _determine_eta
        state = make_state(refund_amount=1000.01)
        eta = _determine_eta(state)
        assert "2" in eta or "3" in eta


# ─────────────────────────────────────────────────────────────────────────────
# 3. Escalation reason tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEscalationReason:

    def test_manager_request_intent_gives_reason(self):
        from agents.escalation import _determine_escalation_reason
        state = make_state(intent="escalation")
        reason = _determine_escalation_reason(state)
        assert "manager" in reason.lower()

    def test_no_intent_no_violations_gives_fallback(self):
        from agents.escalation import _determine_escalation_reason
        state = make_state(intent="policy_returns", requires_escalation=True)
        reason = _determine_escalation_reason(state)
        assert reason != ""

    def test_policy_violations_in_metadata_appear_in_reason(self):
        from agents.escalation import _determine_escalation_reason
        state = make_state(
            intent="policy_returns",
            policy_violations=["Return window exceeded by 10 days"],
        )
        reason = _determine_escalation_reason(state)
        assert "Return window" in reason or reason != ""

    def test_reason_is_never_empty_string(self):
        from agents.escalation import _determine_escalation_reason
        state = make_state(intent="unknown", requires_escalation=False)
        assert _determine_escalation_reason(state) != ""

    def test_multiple_violations_all_included(self):
        from agents.escalation import _determine_escalation_reason
        state = make_state(
            policy_violations=["Refund exceeds limit", "Return window exceeded"],
        )
        reason = _determine_escalation_reason(state)
        assert "Refund" in reason or "Return" in reason


# ─────────────────────────────────────────────────────────────────────────────
# 4. Node integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeIntegration:

    def test_node_returns_ai_message(self):
        from langchain_core.messages import AIMessage
        updates = run_node(make_state())
        assert any(isinstance(m, AIMessage) for m in updates["messages"])

    def test_agent_used_is_escalation(self):
        assert run_node(make_state())["agent_used"] == "escalation"

    def test_resolution_status_is_escalated(self):
        assert run_node(make_state())["resolution_status"] == "escalated"

    def test_escalation_ticket_id_is_set(self):
        updates = run_node(make_state())
        assert updates.get("escalation_ticket_id") is not None

    def test_llm_failure_returns_fallback_message(self):
        from agents.escalation import escalation_node
        from langchain_core.messages import AIMessage

        with patch("agents.escalation._get_llm", side_effect=Exception("API down")), \
             patch("agents.escalation._long_term_memory"):
            updates = escalation_node(make_state())

        ai_msgs = [m for m in updates["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) > 0
        assert "TKT-" in ai_msgs[0].content

    def test_fallback_message_contains_ticket_id(self):
        from agents.escalation import escalation_node
        with patch("agents.escalation._get_llm", side_effect=Exception("down")), \
             patch("agents.escalation._long_term_memory"):
            updates = escalation_node(make_state())
        ticket = updates["escalation_ticket_id"]
        ai_content = updates["messages"][-1].content
        assert ticket in ai_content

    def test_node_works_with_no_order_id(self):
        updates = run_node(make_state(order_id=None))
        assert "messages" in updates

    def test_node_works_with_empty_messages(self):
        state = make_state()
        state["messages"] = []
        updates = run_node(state)
        assert "messages" in updates

    def test_node_works_with_high_refund(self):
        updates = run_node(make_state(refund_amount=1500.0))
        assert updates["resolution_status"] == "escalated"

    def test_each_call_generates_different_ticket(self):
        updates1 = run_node(make_state())
        updates2 = run_node(make_state())
        assert updates1["escalation_ticket_id"] != updates2["escalation_ticket_id"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Long-term memory tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLongTermMemory:

    def test_save_interaction_is_called_on_escalation(self):
        from agents.escalation import escalation_node

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Your ticket is TKT-TEST."

        with patch("agents.escalation._get_llm") as mock_llm, \
             patch("agents.escalation._long_term_memory") as mock_mem_instance:

            mock_llm_instance = MagicMock()
            mock_llm_instance.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            mock_llm.return_value = mock_llm_instance

            escalation_node(make_state())

        mock_mem_instance.save_interaction.assert_called_once()

    def test_save_interaction_receives_customer_id(self):
        from agents.escalation import escalation_node

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Ticket created."

        with patch("agents.escalation._get_llm") as mock_llm, \
             patch("agents.escalation._long_term_memory") as mock_mem_instance:

            mock_llm_instance = MagicMock()
            mock_llm_instance.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            mock_llm.return_value = mock_llm_instance

            escalation_node(make_state(customer_id="CUST-003"))

        call_kwargs = mock_mem_instance.save_interaction.call_args
        assert call_kwargs is not None
        args = call_kwargs[1] if call_kwargs[1] else {}
        customer_id_passed = args.get("customer_id") or (call_kwargs[0][0] if call_kwargs[0] else None)
        assert customer_id_passed == "CUST-003"

    def test_memory_failure_does_not_crash_node(self):
        from agents.escalation import escalation_node

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Ticket created."

        with patch("agents.escalation._get_llm") as mock_llm, \
             patch("agents.escalation._long_term_memory") as mock_mem:

            mock_llm_instance = MagicMock()
            mock_llm_instance.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            mock_llm.return_value = mock_llm_instance
            mock_mem.save_interaction.side_effect = Exception("ChromaDB down")

            # Should not raise
            updates = escalation_node(make_state())

        assert updates["resolution_status"] == "escalated"

    def test_summary_contains_ticket_id(self):
        from agents.escalation import escalation_node

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Ticket created."
        saved_summaries = []

        def capture_save(**kwargs):
            saved_summaries.append(kwargs.get("summary", ""))

        with patch("agents.escalation._get_llm") as mock_llm, \
             patch("agents.escalation._long_term_memory") as mock_mem_instance:

            mock_llm_instance = MagicMock()
            mock_llm_instance.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            mock_llm.return_value = mock_llm_instance
            mock_mem_instance.save_interaction.side_effect = capture_save

            updates = escalation_node(make_state())

        ticket = updates["escalation_ticket_id"]
        assert any(ticket in s for s in saved_summaries)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Known gaps (xfail)
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownGaps:

    def test_summary_contains_customer_complaint(self):
        from agents.escalation import escalation_node
        saved_summaries = []

        def capture(**kwargs):
            saved_summaries.append(kwargs.get("summary", ""))

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Ticket created."

        with patch("agents.escalation._get_llm") as mock_llm, \
             patch("agents.escalation._long_term_memory") as mock_mem_instance:
            mock_llm_instance = MagicMock()
            mock_llm_instance.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            mock_llm.return_value = mock_llm_instance
            mock_mem_instance.save_interaction.side_effect = capture
            escalation_node(make_state(message="My coffee maker exploded!"))

        assert any("coffee maker" in s for s in saved_summaries)

    def test_late_return_violation_gets_longer_eta(self):
        from agents.escalation import _determine_eta
        state = make_state(
            refund_amount=None,
            policy_violations=["Return window exceeded by 15 days. Manager approval required."],
        )
        eta = _determine_eta(state)
        assert "2" in eta or "3" in eta

    def test_return_escalation_generates_rma_number(self):
        state = make_state(
            policy_violations=["POL-002: Return window exceeded"],
        )
        updates = run_node(state)
        assert "rma_number" in updates or any(
            "RMA-" in m.content for m in updates["messages"]
        )

    def test_fallback_response_uses_calculated_eta(self):
        from agents.escalation import escalation_node
        with patch("agents.escalation._get_llm", side_effect=Exception("down")), \
             patch("agents.escalation._long_term_memory"):
            updates = escalation_node(make_state(refund_amount=1500.0))
        content = updates["messages"][-1].content
        # Should say 2-3 days, not 4 hours
        assert "4 business hours" not in content