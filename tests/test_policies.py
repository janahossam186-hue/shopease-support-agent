"""
Policy agent test suite — policy_returns.py & policy_guardrail.py

Sections:
  1. PolicyGuardrail unit tests   — pure rule engine, no LLM, no API key
  2. Node integration tests       — policy_returns_node() state machine
  3. Policy gap tests             — xfail for rules in the doc but missing in code
  4. Edge case tests              — boundary conditions and tricky inputs

Run all:
    pytest tests/test_policies.py -v

Run just guardrail rules (fastest, no mocking):
    pytest tests/test_policies.py -v -k "Guardrail"

Run gap tests (shows what policy doc says vs what code does):
    pytest tests/test_policies.py -v -k "Gaps"
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def days_ago(n: int) -> str:
    """Return an ISO date string n days before today."""
    return (date.today() - timedelta(days=n)).isoformat()


def days_from_now(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()


def make_node_state(
    message: str = "I want to return my item",
    customer_id: str = "CUST-001",
    order_id: str | None = None,
    refund_amount: float | None = None,
) -> dict:
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=message)],
        "customer_id": customer_id,
        "order_id": order_id,
        "refund_amount": refund_amount,
        "metadata": {"past_context": "No prior interactions."},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. PolicyGuardrail unit tests  (pure logic, zero dependencies)
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardrailPOL001RefundAmount:
    """POL-001: self-service refund limit ($500)."""

    def setup_method(self):
        from guardrails.policy_guardrail import PolicyGuardrail
        self.g = PolicyGuardrail()

    def test_none_amount_is_compliant(self):
        r = self.g.check_refund_amount(None)
        assert r.compliant is True
        assert r.requires_escalation is False

    def test_zero_amount_is_non_compliant(self):
        r = self.g.check_refund_amount(0.0)
        assert r.compliant is False
        assert r.requires_escalation is False

    def test_negative_amount_is_non_compliant(self):
        r = self.g.check_refund_amount(-10.0)
        assert r.compliant is False

    def test_amount_below_limit_is_compliant(self):
        r = self.g.check_refund_amount(100.0)
        assert r.compliant is True
        assert r.requires_escalation is False

    def test_amount_at_exact_limit_is_compliant(self):
        r = self.g.check_refund_amount(500.0)
        assert r.compliant is True
        assert r.requires_escalation is False

    def test_amount_one_cent_over_limit_escalates(self):
        r = self.g.check_refund_amount(500.01)
        assert r.compliant is False
        assert r.requires_escalation is True

    def test_large_refund_escalates(self):
        r = self.g.check_refund_amount(1500.0)
        assert r.compliant is False
        assert r.requires_escalation is True

    def test_violation_message_mentions_escalation(self):
        r = self.g.check_refund_amount(600.0)
        assert "supervisor" in r.violation.lower() or "manager" in r.violation.lower()

    def test_rule_id_is_POL001(self):
        assert self.g.check_refund_amount(600.0).rule_id == "POL-001"


class TestGuardrailPOL002ReturnWindow:
    """POL-002: 30-day return window from delivery date."""

    def setup_method(self):
        from guardrails.policy_guardrail import PolicyGuardrail
        self.g = PolicyGuardrail()

    def test_no_date_info_passes(self):
        r = self.g.check_return_window(None, None)
        assert r.compliant is True

    def test_delivered_today_is_within_window(self):
        r = self.g.check_return_window(days_ago(0))
        assert r.compliant is True

    def test_delivered_15_days_ago_is_within_window(self):
        r = self.g.check_return_window(days_ago(15))
        assert r.compliant is True

    def test_delivered_29_days_ago_is_within_window(self):
        r = self.g.check_return_window(days_ago(29))
        assert r.compliant is True

    def test_delivered_30_days_ago_is_within_window(self):
        # exactly 30 days — still within window (boundary inclusive)
        r = self.g.check_return_window(days_ago(30))
        assert r.compliant is True

    def test_delivered_31_days_ago_is_outside_window(self):
        r = self.g.check_return_window(days_ago(31))
        assert r.compliant is False
        assert r.requires_escalation is True

    def test_delivered_90_days_ago_escalates(self):
        r = self.g.check_return_window(days_ago(90))
        assert r.compliant is False
        assert r.requires_escalation is True

    def test_violation_message_mentions_days(self):
        r = self.g.check_return_window(days_ago(45))
        assert "day" in r.violation.lower()

    def test_falls_back_to_created_at_when_no_delivery(self):
        # created 10 days ago, not yet delivered — uses created_at
        r = self.g.check_return_window(None, created_at=days_ago(10))
        assert r.compliant is True

    def test_invalid_date_string_passes_gracefully(self):
        r = self.g.check_return_window("not-a-date")
        assert r.compliant is True  # fail-open on parse error

    def test_iso_datetime_string_is_parsed(self):
        dt = (date.today() - timedelta(days=10)).isoformat() + "T12:00:00"
        r = self.g.check_return_window(dt)
        assert r.compliant is True

    def test_rule_id_is_POL002(self):
        assert self.g.check_return_window(days_ago(40)).rule_id == "POL-002"


class TestGuardrailPOL003NonReturnableItems:
    """POL-003: non-returnable item categories."""

    def setup_method(self):
        from guardrails.policy_guardrail import PolicyGuardrail
        self.g = PolicyGuardrail()

    def test_regular_electronics_are_returnable(self):
        r = self.g.check_item_returnable("electronics", "ProBook Laptop 15")
        assert r.compliant is True

    def test_digital_download_is_not_returnable(self):
        r = self.g.check_item_returnable("digital", "Adobe Photoshop License")
        assert r.compliant is False
        assert r.requires_escalation is False

    def test_software_license_is_not_returnable(self):
        r = self.g.check_item_returnable("software", "Windows 11 Pro License")
        assert r.compliant is False

    def test_gift_card_is_not_returnable(self):
        r = self.g.check_item_returnable("gift card", "$50 ShopEase Gift Card")
        assert r.compliant is False

    def test_personal_care_is_not_returnable(self):
        r = self.g.check_item_returnable("personal care", "Electric Shaver")
        assert r.compliant is False

    def test_perishable_food_is_not_returnable(self):
        r = self.g.check_item_returnable("food", "Organic Coffee Beans")
        assert r.compliant is False

    def test_final_sale_item_is_not_returnable(self):
        r = self.g.check_item_returnable("final sale", "Clearance Jacket")
        assert r.compliant is False

    def test_case_insensitive_matching(self):
        r = self.g.check_item_returnable("DIGITAL", "PDF Download")
        assert r.compliant is False

    def test_rule_id_is_POL003(self):
        assert self.g.check_item_returnable("digital", "eBook").rule_id == "POL-003"


class TestPolicyGuardrailCheckConvenience:
    """Tests for the convenience function policy_guardrail_check()."""

    def test_all_none_returns_compliant(self):
        from guardrails.policy_guardrail import policy_guardrail_check
        result = policy_guardrail_check()
        assert result["policy_compliant"] is True
        assert result["requires_escalation"] is False
        assert result["policy_violations"] == []

    def test_high_refund_triggers_escalation(self):
        from guardrails.policy_guardrail import policy_guardrail_check
        result = policy_guardrail_check(refund_amount=750.0)
        assert result["requires_escalation"] is True
        assert "POL-001" in result["policy_rule_ids"]

    def test_late_return_triggers_escalation(self):
        from guardrails.policy_guardrail import policy_guardrail_check
        result = policy_guardrail_check(delivered_at=days_ago(40))
        assert result["requires_escalation"] is True
        assert "POL-002" in result["policy_rule_ids"]

    def test_non_returnable_product_blocks_without_escalation(self):
        from guardrails.policy_guardrail import policy_guardrail_check
        result = policy_guardrail_check(product_name="PDF License", product_category="digital")
        assert result["policy_compliant"] is False
        assert result["requires_escalation"] is False

    def test_multiple_violations_all_captured(self):
        from guardrails.policy_guardrail import policy_guardrail_check
        result = policy_guardrail_check(
            refund_amount=800.0,
            delivered_at=days_ago(50),
        )
        assert len(result["policy_violations"]) == 2
        assert result["requires_escalation"] is True

    def test_compliant_case_within_window_low_refund(self):
        from guardrails.policy_guardrail import policy_guardrail_check
        result = policy_guardrail_check(
            refund_amount=99.99,
            delivered_at=days_ago(5),
        )
        assert result["policy_compliant"] is True
        assert result["requires_escalation"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Node integration tests  (policy_returns_node state machine)
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeIntegration:
    """Tests for policy_returns_node() with mocked LLM and retriever."""

    def _run_node(self, state: dict, llm_response: str = "Your return has been approved."):
        """Run the node with LLM and RAG fully mocked."""
        from agents.policy_returns import policy_returns_node

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = llm_response

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve.return_value = []
        mock_retriever_instance.format_for_prompt.return_value = "No docs."

        mock_llm_instance = MagicMock()
        mock_llm_instance.__or__ = MagicMock(
            return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
        )

        with patch("rag.retriever.get_retriever", return_value=mock_retriever_instance), \
             patch("agents.policy_returns._get_llm", return_value=mock_llm_instance), \
             patch("agents.policy_returns.get_order_tool") as mock_order:

            mock_order.invoke.return_value = "No specific order referenced."
            return policy_returns_node(state)

    def test_node_returns_ai_message(self):
        from langchain_core.messages import AIMessage
        updates = self._run_node(make_node_state())
        assert any(isinstance(m, AIMessage) for m in updates["messages"])

    def test_agent_used_is_policy_returns(self):
        updates = self._run_node(make_node_state())
        assert updates["agent_used"] == "policy_returns"

    def test_compliant_request_resolves(self):
        state = make_node_state(refund_amount=50.0)
        updates = self._run_node(state)
        assert updates["resolution_status"] == "resolved"
        assert updates["requires_escalation"] is False

    def test_high_refund_escalates(self):
        state = make_node_state(
            message="I want a refund of $800",
            refund_amount=800.0,
        )
        updates = self._run_node(state)
        assert updates["resolution_status"] == "escalated"
        assert updates["requires_escalation"] is True

    def test_policy_compliant_field_is_set(self):
        updates = self._run_node(make_node_state(refund_amount=100.0))
        assert "policy_compliant" in updates

    def test_retrieval_scores_are_returned(self):
        updates = self._run_node(make_node_state())
        assert "retrieval_scores" in updates

    def test_llm_failure_returns_fallback_message(self):
        from agents.policy_returns import policy_returns_node
        from langchain_core.messages import AIMessage

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever.format_for_prompt.return_value = ""

        with patch("rag.retriever.get_retriever", return_value=mock_retriever), \
             patch("agents.policy_returns._get_llm", side_effect=Exception("API down")), \
             patch("agents.policy_returns.get_order_tool") as mock_order:

            mock_order.invoke.return_value = ""
            updates = policy_returns_node(make_node_state())

        ai_msgs = [m for m in updates["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) > 0
        assert "returns@shopease.com" in ai_msgs[0].content or "unable" in ai_msgs[0].content.lower()

    def test_no_order_id_still_runs(self):
        state = make_node_state(order_id=None)
        updates = self._run_node(state)
        assert "messages" in updates

    def test_empty_messages_list_still_runs(self):
        state = {
            "messages": [],
            "customer_id": "CUST-001",
            "order_id": None,
            "refund_amount": None,
            "metadata": {},
        }
        updates = self._run_node(state)
        assert "messages" in updates


# ─────────────────────────────────────────────────────────────────────────────
# 3. Edge case tests  (boundary conditions)
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def setup_method(self):
        from guardrails.policy_guardrail import PolicyGuardrail
        self.g = PolicyGuardrail()

    def test_order_total_with_comma_format(self):
        """_extract_order_for_policy should handle $1,299.00 formatted totals."""
        from agents.policy_returns import _extract_order_for_policy
        order_str = "Total: $1,299.00\nDelivered: 2025-05-01\nCreated: 2025-04-28"
        result = _extract_order_for_policy(order_str)
        assert result.get("order_total") == 1299.0

    def test_order_total_simple_format(self):
        from agents.policy_returns import _extract_order_for_policy
        order_str = "Total: $129.99\nDelivered: 2025-05-01"
        result = _extract_order_for_policy(order_str)
        assert result.get("order_total") == 129.99

    def test_extract_delivered_at(self):
        from agents.policy_returns import _extract_order_for_policy
        order_str = "Delivered: 2025-05-10\nCreated: 2025-05-01"
        result = _extract_order_for_policy(order_str)
        assert result.get("delivered_at") == "2025-05-10"

    def test_extract_created_at(self):
        from agents.policy_returns import _extract_order_for_policy
        order_str = "Created: 2025-04-20\nTotal: $99.99"
        result = _extract_order_for_policy(order_str)
        assert result.get("created_at") == "2025-04-20"

    def test_return_window_boundary_exactly_30_days(self):
        r = self.g.check_return_window(days_ago(30))
        assert r.compliant is True

    def test_return_window_boundary_exactly_31_days(self):
        r = self.g.check_return_window(days_ago(31))
        assert r.compliant is False

    def test_refund_amount_exactly_500(self):
        from guardrails.policy_guardrail import PolicyGuardrail
        r = PolicyGuardrail().check_refund_amount(500.0)
        assert r.compliant is True

    def test_refund_amount_500_01_escalates(self):
        from guardrails.policy_guardrail import PolicyGuardrail
        r = PolicyGuardrail().check_refund_amount(500.01)
        assert r.requires_escalation is True

    def test_product_name_with_mixed_case_digital(self):
        r = self.g.check_item_returnable("Electronics", "Digital Download Code")
        assert r.compliant is False

    def test_empty_product_name_skips_POL003(self):
        from guardrails.policy_guardrail import policy_guardrail_check
        result = policy_guardrail_check(product_name="", product_category="digital")
        # POL-003 should not run with empty product_name
        assert "POL-003" not in result.get("policy_rule_ids", [])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Policy gap tests  (xfail — in the policy doc but missing in the code)
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyGaps:
    """
    These tests document rules that exist in returns_policy.md
    but are NOT yet implemented in code. All are marked xfail.

    When a test starts XPASS, the rule has been implemented — remove xfail.
    """

    def setup_method(self):
        from guardrails.policy_guardrail import PolicyGuardrail
        self.g = PolicyGuardrail()

    def test_holiday_return_window_december_purchase(self):
        """Items bought Dec 15 should be returnable until Jan 31 next year."""
        from guardrails.policy_guardrail import policy_guardrail_check
        import datetime
        dec_15 = f"{date.today().year - 1}-12-15"
        jan_20 = f"{date.today().year}-01-20"
        # Simulate: purchased Dec 15, trying to return Jan 20 — should be compliant
        with patch("guardrails.policy_guardrail.date") as mock_date:
            mock_date.today.return_value = date.fromisoformat(jan_20)
            mock_date.fromisoformat = date.fromisoformat
            r = self.g.check_return_window(None, created_at=dec_15)
        assert r.compliant is True

    def test_mid_tier_refund_gets_supervisor_not_manager(self):
        """$750 refund should be supervisor tier (1 day), not manager tier (2-3 days)."""
        r = self.g.check_refund_amount(750.0)
        assert hasattr(r, "escalation_tier")
        assert r.escalation_tier == "supervisor"

    def test_high_refund_gets_manager_tier(self):
        """$1500 refund should be manager tier (2-3 days)."""
        r = self.g.check_refund_amount(1500.0)
        assert hasattr(r, "escalation_tier")
        assert r.escalation_tier == "manager"

    def test_damage_reported_after_48_hours_requires_review(self):
        """Defective item reported 5 days after delivery should require review."""
        from guardrails.policy_guardrail import policy_guardrail_check
        result = policy_guardrail_check(
            delivered_at=days_ago(5),
            damage_reported_at=days_ago(0),
        )
        assert result.get("late_damage_report") is True

    def test_late_return_result_includes_restocking_fee(self):
        """Return 40 days after delivery should flag the 15% restocking fee."""
        r = self.g.check_return_window(days_ago(40))
        assert hasattr(r, "restocking_fee_pct")
        assert r.restocking_fee_pct == 15

    def test_undelivered_order_blocks_return(self):
        """Order not yet delivered (no delivered_at) should not allow a return to start."""
        from guardrails.policy_guardrail import policy_guardrail_check
        result = policy_guardrail_check(
            delivered_at=None,
            created_at=days_ago(3),
        )
        assert result["policy_compliant"] is False

    def test_return_dropoff_window_expired(self):
        """Return initiated 10 days ago with no drop-off should be flagged."""
        from guardrails.policy_guardrail import policy_guardrail_check
        result = policy_guardrail_check(
            return_initiated_at=days_ago(10),
        )
        assert result.get("dropoff_window_expired") is True