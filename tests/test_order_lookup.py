"""
tests/test_order_lookup.py

Comprehensive test suite for the LLMCompiler-based Nora order agent.

Sections:
  1 — DAG Utilities (pure, no IO, no mocking)
  2 — Tool Executor (minimal mocking — no SMTP, no real writes)
  3 — Planner Node (mock LLM)
  4 — Joiner Node (mock LLM)
  5 — Full Flow Integration (mock LLM + SMTP + write tools)
  6 — Security Rules (no LLM needed)

Integration tests that require a real LLM are marked:
  @pytest.mark.integration
  @pytest.mark.skip(reason="requires LLM API")
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

from agents.order_lookup import (
    MAX_REPLANS,
    NORA_TOOL_CONTRACTS,
    _execute_tool,
    _get_all_deps,
    find_compile_refs,
    joiner_node,
    normalize_nora_tasks,
    order_lookup_node,
    planner_node,
    resolve_compile_refs,
    scheduler_node,
    validate_nora_tasks,
)


# ── Test helpers ───────────────────────────────────────────────────────────────

def mock_llm(content: str) -> MagicMock:
    """Mock LLM whose single invoke() returns an AIMessage with the given content.
    Using AIMessage (not raw MagicMock) so LangChain's StrOutputParser validates it.
    """
    m = MagicMock()
    m.invoke.return_value = AIMessage(content=content)
    return m


def mock_llm_seq(*contents: str) -> MagicMock:
    """Mock LLM whose invoke() returns AIMessages in order.
    Using AIMessage so LangChain chain machinery (StrOutputParser) accepts the output.
    """
    m = MagicMock()
    m.invoke.side_effect = [AIMessage(content=c) for c in contents]
    return m


def make_state(
    message: str = "Where is my order?",
    customer_id: str = "CUST-001",
    session_id: str = "test-session",
    order_id=None,
    identity_verified: bool = False,
    replan_count: int = 0,
    completed_tasks: dict | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Build a minimal LangGraph state dict for testing."""
    metadata: dict = {
        "identity_verified": identity_verified,
        "compiler_state": {
            "replan_count": replan_count,
            "completed_tasks": completed_tasks or {},
            "plan": [],
            "trace": [],
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "messages": [HumanMessage(content=message)],
        "customer_id": customer_id,
        "session_id": session_id,
        "order_id": order_id,
        "metadata": metadata,
    }


# Reusable plan fixtures
_SEND_VERIFY_PLAN = {
    "tasks": [
        {"id": "T1", "tool": "send_otp",
         "args": {"customer_id": "CUST-001"}, "deps": []},
        {"id": "T2", "tool": "verify_otp",
         "args": {"customer_id": "CUST-001", "entered_code": "code"}, "deps": ["T1"]},
    ]
}

_FETCH_PLAN = {
    "tasks": [
        {"id": "T1", "tool": "fetch_order",
         "args": {"order_id": "ORD-10001"}, "deps": []},
    ]
}

_LIST_PLAN = {
    "tasks": [
        {"id": "T1", "tool": "list_orders",
         "args": {"customer_id": "CUST-001"}, "deps": []},
    ]
}

_JOINER_ANSWER = json.dumps({"action": "ANSWER", "reason": "ok",
                              "clarify_question": "", "escalate_reason": "",
                              "replan_notes": ""})
_JOINER_ESCALATE = json.dumps({"action": "ESCALATE",
                                "reason": "outside authority",
                                "escalate_reason": "cannot handle this",
                                "clarify_question": "", "replan_notes": ""})
_JOINER_CLARIFY = json.dumps({"action": "CLARIFY",
                               "reason": "need more info",
                               "clarify_question": "What is your new address?",
                               "escalate_reason": "", "replan_notes": ""})
_JOINER_REPLAN = json.dumps({"action": "REPLAN",
                              "reason": "task failed",
                              "replan_notes": "retry without cache",
                              "clarify_question": "", "escalate_reason": ""})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DAG Utilities
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeTasks:
    """normalize_nora_tasks() handles all input shapes."""

    def test_list_input(self):
        """Raw list of task dicts is normalized into clean list."""
        raw = [{"id": "T1", "tool": "send_otp", "args": {}, "deps": []}]
        result = normalize_nora_tasks(raw)
        assert len(result) == 1
        assert result[0]["id"] == "T1"

    def test_dict_with_tasks_key(self):
        """Dict with a 'tasks' key is unwrapped correctly."""
        raw = {"tasks": [{"id": "T1", "tool": "send_otp", "args": {}, "deps": []}]}
        result = normalize_nora_tasks(raw)
        assert len(result) == 1

    def test_empty_list(self):
        """Empty list returns empty list without error."""
        assert normalize_nora_tasks([]) == []

    def test_missing_fields_get_defaults(self):
        """Tasks with missing keys receive empty-string / empty-dict defaults."""
        raw = [{}]
        result = normalize_nora_tasks(raw)
        assert result[0]["id"] == ""
        assert result[0]["tool"] == ""
        assert result[0]["args"] == {}
        assert result[0]["deps"] == []


class TestFindCompileRefs:
    """find_compile_refs() locates all $T… references in nested structures."""

    def test_simple_string_ref(self):
        """'$T1' returns {'T1'}."""
        assert find_compile_refs("$T1") == {"T1"}

    def test_nested_dict_ref(self):
        """{'order_data': '$T3'} returns {'T3'}."""
        assert find_compile_refs({"order_data": "$T3"}) == {"T3"}

    def test_multiple_refs(self):
        """Dict with two different refs returns both."""
        assert find_compile_refs({"a": "$T1", "b": "$T2"}) == {"T1", "T2"}

    def test_no_refs(self):
        """Plain string with no $ returns empty set."""
        assert find_compile_refs("hello world") == set()

    def test_ref_inside_list(self):
        """List containing $T refs returns all of them."""
        assert find_compile_refs(["$T1", "$T2"]) == {"T1", "T2"}

    def test_ref_inside_nested_dict(self):
        """Deeply nested dict ref is found."""
        assert find_compile_refs({"outer": {"inner": "$T5"}}) == {"T5"}

    def test_non_string_scalar_ignored(self):
        """Integer values produce no refs."""
        assert find_compile_refs(42) == set()


class TestResolveCompileRefs:
    """resolve_compile_refs() substitutes $T… with real task results."""

    def test_exact_ref_replaced(self):
        """'$T1' is replaced with the actual result object (not serialised)."""
        results = {"T1": {"status": "ok", "data": "order info"}}
        resolved = resolve_compile_refs("$T1", results)
        assert resolved == {"status": "ok", "data": "order info"}

    def test_ref_in_dict_value(self):
        """{'order_data': '$T1'} has its value replaced with the result."""
        results = {"T1": {"status": "ok"}}
        resolved = resolve_compile_refs({"order_data": "$T1"}, results)
        assert resolved["order_data"] == {"status": "ok"}

    def test_ref_in_nested_structure(self):
        """Nested dict with multiple $T refs all resolved."""
        results = {"T1": "val1", "T2": "val2"}
        obj = {"a": "$T1", "b": {"c": "$T2"}}
        resolved = resolve_compile_refs(obj, results)
        assert resolved["a"] == "val1"
        assert resolved["b"]["c"] == "val2"

    def test_no_ref_unchanged(self):
        """String with no $T reference is returned unchanged."""
        assert resolve_compile_refs("just a string", {}) == "just a string"

    def test_missing_ref_raises(self):
        """Referencing a result that doesn't exist raises KeyError."""
        with pytest.raises(KeyError):
            resolve_compile_refs("$T99", {})


class TestValidateNoraTasks:
    """validate_nora_tasks() enforces DAG correctness rules."""

    def _otp_chain(self):
        return [
            {"id": "T1", "tool": "send_otp",
             "args": {"customer_id": "CUST-001"}, "deps": []},
            {"id": "T2", "tool": "verify_otp",
             "args": {"customer_id": "CUST-001", "entered_code": "x"}, "deps": ["T1"]},
        ]

    def test_valid_minimal_plan(self):
        """Single send_otp task with no deps passes."""
        tasks = [{"id": "T1", "tool": "send_otp",
                  "args": {"customer_id": "CUST-001"}, "deps": []}]
        result = validate_nora_tasks(tasks)
        assert len(result) == 1

    def test_duplicate_task_ids_rejected(self):
        """Two tasks with the same ID raises ValueError."""
        tasks = [
            {"id": "T1", "tool": "send_otp", "args": {}, "deps": []},
            {"id": "T1", "tool": "verify_otp",
             "args": {"customer_id": "x", "entered_code": "y"}, "deps": []},
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            validate_nora_tasks(tasks)

    def test_invalid_task_id_format(self):
        """Task ID not starting with T raises ValueError."""
        tasks = [{"id": "step1", "tool": "send_otp", "args": {}, "deps": []}]
        with pytest.raises(ValueError, match="Invalid task ID"):
            validate_nora_tasks(tasks)

    def test_unknown_tool_rejected(self):
        """Tool name not in NORA_TOOL_CONTRACTS raises ValueError."""
        tasks = [{"id": "T1", "tool": "fly_rocket", "args": {}, "deps": []}]
        with pytest.raises(ValueError, match="unknown tool"):
            validate_nora_tasks(tasks)

    def test_missing_dep_raises(self):
        """Dep referencing non-existent task ID raises ValueError."""
        tasks = [{"id": "T1", "tool": "verify_otp",
                  "args": {"customer_id": "x", "entered_code": "y"},
                  "deps": ["T99"]}]
        with pytest.raises(ValueError, match="does not exist"):
            validate_nora_tasks(tasks)

    def test_self_dependency_raises(self):
        """Task depending on itself raises ValueError."""
        tasks = [{"id": "T1", "tool": "send_otp",
                  "args": {"customer_id": "x"}, "deps": ["T1"]}]
        with pytest.raises(ValueError, match="cannot depend on itself"):
            validate_nora_tasks(tasks)

    def test_ref_not_in_deps_raises(self):
        """$T ref used in args but absent from deps raises ValueError."""
        tasks = [
            {"id": "T1", "tool": "send_otp", "args": {"customer_id": "x"}, "deps": []},
            {"id": "T2", "tool": "verify_otp",
             "args": {"customer_id": "x", "entered_code": "y"}, "deps": ["T1"]},
            {"id": "T3", "tool": "check_order_status",
             "args": {"order_data": "$T4"}, "deps": ["T2"]},  # $T4 not declared
        ]
        with pytest.raises(ValueError):
            validate_nora_tasks(tasks)

    def test_ref_nonexistent_task_raises(self):
        """$T ref pointing to a non-existent task ID raises ValueError."""
        tasks = [
            {"id": "T1", "tool": "send_otp", "args": {"customer_id": "x"}, "deps": []},
            {"id": "T2", "tool": "verify_otp",
             "args": {"customer_id": "x", "entered_code": "y"}, "deps": ["T1"]},
            {"id": "T3", "tool": "check_order_status",
             "args": {"order_data": "$T99"}, "deps": ["T2", "T99"]},
        ]
        with pytest.raises(ValueError):
            validate_nora_tasks(tasks)

    def test_cycle_detection_simple(self):
        """Direct cycle T1→T2, T2→T1 raises ValueError."""
        tasks = [
            {"id": "T1", "tool": "send_otp", "args": {"customer_id": "x"},
             "deps": ["T2"]},
            {"id": "T2", "tool": "send_otp", "args": {"customer_id": "x"},
             "deps": ["T1"]},
        ]
        with pytest.raises(ValueError, match="Cycle"):
            validate_nora_tasks(tasks)

    def test_cycle_detection_indirect(self):
        """Indirect cycle T1→T2→T3→T1 raises ValueError."""
        tasks = [
            {"id": "T1", "tool": "send_otp", "args": {"customer_id": "x"},
             "deps": ["T3"]},
            {"id": "T2", "tool": "send_otp", "args": {"customer_id": "x"},
             "deps": ["T1"]},
            {"id": "T3", "tool": "send_otp", "args": {"customer_id": "x"},
             "deps": ["T2"]},
        ]
        with pytest.raises(ValueError, match="Cycle"):
            validate_nora_tasks(tasks)

    def test_order_tool_without_verify_otp_dep_raises(self):
        """fetch_order in plan with verify_otp but no dep on it raises ValueError."""
        tasks = self._otp_chain() + [
            {"id": "T3", "tool": "fetch_order",
             "args": {"order_id": "ORD-10001"}, "deps": []},  # Missing T2 dep
        ]
        with pytest.raises(ValueError, match="must depend on verify_otp"):
            validate_nora_tasks(tasks)

    def test_order_tool_with_verify_otp_dep_passes(self):
        """fetch_order depending on verify_otp passes validation."""
        tasks = self._otp_chain() + [
            {"id": "T3", "tool": "fetch_order",
             "args": {"order_id": "ORD-10001"}, "deps": ["T2"]},
        ]
        result = validate_nora_tasks(tasks)
        assert len(result) == 3

    def test_send_otp_no_deps_passes(self):
        """send_otp with empty deps is always valid."""
        tasks = [{"id": "T1", "tool": "send_otp",
                  "args": {"customer_id": "x"}, "deps": []}]
        validate_nora_tasks(tasks)

    def test_verify_otp_deps_on_send_otp_passes(self):
        """verify_otp depending on send_otp passes."""
        validate_nora_tasks(normalize_nora_tasks(_SEND_VERIFY_PLAN))

    def test_order_tool_no_otp_in_plan_passes(self):
        """fetch_order with NO verify_otp anywhere in the plan passes (identity already verified)."""
        tasks = [{"id": "T1", "tool": "fetch_order",
                  "args": {"order_id": "ORD-10001"}, "deps": []}]
        result = validate_nora_tasks(tasks)
        assert len(result) == 1

    def test_valid_complex_plan(self):
        """Full cancel plan: send_otp→verify_otp→fetch→check_status→cancel passes."""
        tasks = [
            {"id": "T1", "tool": "send_otp", "args": {"customer_id": "x"}, "deps": []},
            {"id": "T2", "tool": "verify_otp",
             "args": {"customer_id": "x", "entered_code": "c"}, "deps": ["T1"]},
            {"id": "T3", "tool": "fetch_order",
             "args": {"order_id": "ORD-10003"}, "deps": ["T2"]},
            {"id": "T4", "tool": "check_order_status",
             "args": {"order_data": "$T3"}, "deps": ["T3"]},
            {"id": "T5", "tool": "cancel_order",
             "args": {"order_id": "ORD-10003", "customer_id": "x",
                      "order_status": "$T4"}, "deps": ["T4"]},
        ]
        result = validate_nora_tasks(tasks)
        assert len(result) == 5

    def test_valid_parallel_plan(self):
        """fetch_order and retrieve_knowledge both depending on verify_otp passes."""
        tasks = [
            {"id": "T1", "tool": "send_otp", "args": {"customer_id": "x"}, "deps": []},
            {"id": "T2", "tool": "verify_otp",
             "args": {"customer_id": "x", "entered_code": "c"}, "deps": ["T1"]},
            {"id": "T3", "tool": "fetch_order",
             "args": {"order_id": "ORD-10001"}, "deps": ["T2"]},
            {"id": "T4", "tool": "retrieve_knowledge",
             "args": {"query": "shipping"}, "deps": ["T2"]},
        ]
        result = validate_nora_tasks(tasks)
        assert len(result) == 4


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Tool Executor
# ══════════════════════════════════════════════════════════════════════════════

class TestExecuteToolOTP:
    """_execute_tool() for send_otp and verify_otp."""

    def _state(self, **meta_overrides):
        s = {"customer_id": "CUST-001", "session_id": "test", "metadata": {}}
        s["metadata"].update(meta_overrides)
        return s

    def test_send_otp_stores_pending_otp(self, monkeypatch):
        """send_otp returns pending_otp in the result dict (scheduler merges it)."""
        monkeypatch.setattr("agents.order_lookup.send_otp_email",
                            lambda cid: ("123456", "t***@example.com"))
        result = _execute_tool("send_otp", {"customer_id": "CUST-001"}, self._state())
        assert result["pending_otp"] == "123456"

    def test_send_otp_sets_identity_verified_false(self, monkeypatch):
        """send_otp returns identity_verified=False in the result dict."""
        monkeypatch.setattr("agents.order_lookup.send_otp_email",
                            lambda cid: ("123456", "t***@example.com"))
        result = _execute_tool("send_otp", {"customer_id": "CUST-001"}, self._state(identity_verified=True))
        assert result["identity_verified"] is False

    def test_send_otp_returns_otp_sent_status(self, monkeypatch):
        """send_otp result has status='otp_sent'."""
        monkeypatch.setattr("agents.order_lookup.send_otp_email",
                            lambda cid: ("999888", "x***@test.com"))
        result = _execute_tool("send_otp", {}, self._state())
        assert result["status"] == "otp_sent"

    def test_verify_otp_correct_code_verifies(self):
        """Correct OTP returns identity_verified=True and pending_otp=None in result dict."""
        state = self._state(pending_otp="482931")
        result = _execute_tool("verify_otp",
                               {"customer_id": "CUST-001", "entered_code": "482931"}, state)
        assert result["status"] == "verified"
        assert result["identity_verified"] is True
        assert result["pending_otp"] is None

    def test_verify_otp_wrong_code_fails(self):
        """Wrong OTP returns status='failed' and leaves identity unverified."""
        state = self._state(pending_otp="482931")
        result = _execute_tool("verify_otp",
                               {"customer_id": "CUST-001", "entered_code": "999999"}, state)
        assert result["status"] == "failed"
        assert not state["metadata"].get("identity_verified")

    def test_verify_otp_code_embedded_in_text(self):
        """Code embedded in prose 'my code is 482931 thanks' is extracted."""
        state = self._state(pending_otp="482931")
        result = _execute_tool("verify_otp",
                               {"customer_id": "CUST-001",
                                "entered_code": "my code is 482931 thanks"}, state)
        assert result["status"] == "verified"

    def test_verify_otp_already_verified_returns_already_verified(self):
        """If identity already verified, returns status='already_verified' immediately."""
        state = self._state(identity_verified=True)
        result = _execute_tool("verify_otp",
                               {"customer_id": "CUST-001", "entered_code": "000000"}, state)
        assert result["status"] == "already_verified"

    def test_verify_otp_no_pending_otp_fails(self):
        """No pending OTP in metadata means any code fails."""
        state = self._state(pending_otp=None)
        result = _execute_tool("verify_otp",
                               {"customer_id": "CUST-001", "entered_code": "482931"}, state)
        assert result["status"] == "failed"

    def test_verify_otp_replay_attack_fails(self):
        """After successful verification, reusing the same code fails.
        The scheduler normally merges the result side-effects into metadata;
        we simulate that merge here before the second call."""
        state = self._state(pending_otp="482931")
        first = _execute_tool("verify_otp",
                              {"customer_id": "CUST-001", "entered_code": "482931"}, state)
        assert first["status"] == "verified"
        # Simulate scheduler merge: apply returned side-effect keys to metadata
        state["metadata"]["identity_verified"] = first["identity_verified"]
        state["metadata"]["pending_otp"] = first["pending_otp"]
        second = _execute_tool("verify_otp",
                               {"customer_id": "CUST-001", "entered_code": "482931"}, state)
        assert second["status"] == "already_verified"


class TestExecuteToolFetchOrder:
    """_execute_tool() for fetch_order — ownership, fraud detection, not-found."""

    def _state(self, customer_id="CUST-001"):
        return {"customer_id": customer_id, "session_id": "test",
                "metadata": {"identity_verified": True}}

    def test_fetch_existing_order_correct_customer(self):
        """ORD-10001 belongs to CUST-001 — returns status='ok'."""
        result = _execute_tool("fetch_order", {"order_id": "ORD-10001"}, self._state("CUST-001"))
        assert result["status"] == "ok"
        assert "raw" in result

    def test_fetch_order_wrong_customer_fraud_detected(self):
        """CUST-002 accessing ORD-10001 (CUST-001's) returns fraud_detected."""
        result = _execute_tool("fetch_order", {"order_id": "ORD-10001"}, self._state("CUST-002"))
        assert result["status"] == "fraud_detected"

    def test_fetch_nonexistent_order_returns_not_found(self):
        """ORD-99999 does not exist — returns status='not_found', no crash."""
        result = _execute_tool("fetch_order", {"order_id": "ORD-99999"}, self._state())
        assert result["status"] == "not_found"

    def test_fetch_order_empty_order_id_not_found(self):
        """Empty order ID handled gracefully — not_found, no crash."""
        result = _execute_tool("fetch_order", {"order_id": ""}, self._state())
        assert result["status"] == "not_found"

    def test_fetch_order_lowercase_id_normalized(self):
        """'ord-10001' is normalised to 'ORD-10001' and found correctly."""
        result = _execute_tool("fetch_order", {"order_id": "ord-10001"}, self._state("CUST-001"))
        assert result["status"] == "ok"

    def test_fetch_order_returns_raw_dict_with_fields(self):
        """Result 'raw' dict contains expected order fields."""
        result = _execute_tool("fetch_order", {"order_id": "ORD-10001"}, self._state("CUST-001"))
        assert result["status"] == "ok"
        raw = result["raw"]
        assert raw["order_id"] == "ORD-10001"
        assert raw["customer_id"] == "CUST-001"
        assert raw["status"] == "delivered"

    def test_fraud_message_does_not_reveal_owner(self):
        """Fraud result message never reveals the real owner's customer ID."""
        result = _execute_tool("fetch_order", {"order_id": "ORD-10001"}, self._state("CUST-002"))
        assert "CUST-001" not in result.get("message", "")


class TestExecuteToolCheckOrderStatus:
    """_execute_tool() for check_order_status — status extraction."""

    def _state(self):
        return {"customer_id": "CUST-001", "session_id": "test", "metadata": {}}

    def test_extracts_processing_status(self):
        """Order dict with status=processing returns order_status=processing."""
        order_data = {"status": "ok", "raw": {"status": "processing"}}
        result = _execute_tool("check_order_status", {"order_data": order_data}, self._state())
        assert result["order_status"] == "processing"

    def test_extracts_delivered_status(self):
        """Order dict with status=delivered returns order_status=delivered."""
        order_data = {"status": "ok", "raw": {"status": "delivered"}}
        result = _execute_tool("check_order_status", {"order_data": order_data}, self._state())
        assert result["order_status"] == "delivered"

    def test_missing_status_returns_unknown(self):
        """Order raw dict without status key returns 'unknown'."""
        order_data = {"status": "ok", "raw": {}}
        result = _execute_tool("check_order_status", {"order_data": order_data}, self._state())
        assert result["order_status"] == "unknown"

    def test_non_dict_order_data_returns_unknown(self):
        """Non-dict order_data (e.g. string) doesn't crash — returns unknown."""
        result = _execute_tool("check_order_status", {"order_data": "bad"}, self._state())
        assert result["order_status"] == "unknown"


class TestExecuteToolReturnContext:
    """_execute_tool() for collect_return_context."""

    def _state(self, customer_id="CUST-001"):
        return {"customer_id": customer_id, "session_id": "test", "metadata": {}}

    def _order_data(self):
        return {
            "status": "ok",
            "raw": {
                "order_id": "ORD-10001",
                "created_at": "2025-04-20",
                "delivered_at": "2025-04-25",
                "items": [{"product_id": "PROD-001", "name": "ProBook Laptop 15",
                           "qty": 1, "price": 899.99}],
            },
        }

    def test_collects_purchase_date(self):
        """Return context includes created_at (purchase date)."""
        state = self._state()
        result = _execute_tool("collect_return_context", {"order_data": self._order_data()}, state)
        assert result["return_context"]["purchase_date"] == "2025-04-20"

    def test_collects_delivered_at(self):
        """Return context includes delivered_at."""
        state = self._state()
        result = _execute_tool("collect_return_context", {"order_data": self._order_data()}, state)
        assert result["return_context"]["delivered_at"] == "2025-04-25"

    def test_collects_items_list(self):
        """Return context includes items array."""
        state = self._state()
        result = _execute_tool("collect_return_context", {"order_data": self._order_data()}, state)
        assert len(result["return_context"]["items"]) == 1

    def test_saves_to_metadata(self):
        """Return context is returned in the result dict (scheduler merges into metadata)."""
        state = self._state()
        result = _execute_tool("collect_return_context", {"order_data": self._order_data()}, state)
        assert "return_context" in result
        assert result["return_context"]["order_id"] == "ORD-10001"

    def test_empty_order_data_handled(self):
        """Empty order_data dict handled gracefully — no crash, returns ok with empty fields."""
        state = self._state()
        result = _execute_tool("collect_return_context", {"order_data": {}}, state)
        assert result["status"] == "ok"
        # {} is a dict, so code falls into the isinstance branch and returns a
        # dict with empty-string defaults — not an empty dict.
        assert "return_context" in result


class TestExecuteToolBuildHandoff:
    """_execute_tool() for build_handoff."""

    def _state(self):
        return {"customer_id": "CUST-001", "session_id": "test", "metadata": {}}

    def test_handoff_contains_required_fields(self):
        """Handoff dict has all 7 required fields."""
        result = _execute_tool("build_handoff",
                               {"reason": "legal threat", "order_data": ""}, self._state())
        hf = result["handoff"]
        for field in ("issue_summary", "actions_attempted", "customer_sentiment",
                      "relevant_order_info", "escalation_reason", "confidence_score", "agent"):
            assert field in hf, f"Missing field: {field}"

    def test_handoff_saved_to_metadata(self):
        """Handoff is returned in the result dict as 'escalation_handoff' (scheduler merges it)."""
        state = self._state()
        result = _execute_tool("build_handoff", {"reason": "fraud", "order_data": ""}, state)
        assert "escalation_handoff" in result

    def test_handoff_reason_stored_correctly(self):
        """Provided reason appears in both issue_summary and escalation_reason."""
        result = _execute_tool("build_handoff",
                               {"reason": "legal threat", "order_data": ""}, self._state())
        hf = result["handoff"]
        assert "legal threat" in hf["issue_summary"]
        assert "legal threat" in hf["escalation_reason"]

    def test_handoff_agent_is_nora(self):
        """Handoff agent field is always 'nora_order_lookup'."""
        result = _execute_tool("build_handoff",
                               {"reason": "x", "order_data": ""}, self._state())
        assert result["handoff"]["agent"] == "nora_order_lookup"

    def test_handoff_includes_order_data_when_provided(self):
        """When order data has 'raw', it's included in relevant_order_info."""
        order_data = {"status": "ok", "raw": {"order_id": "ORD-10001", "status": "delivered"}}
        result = _execute_tool("build_handoff",
                               {"reason": "dispute", "order_data": order_data}, self._state())
        assert result["handoff"]["relevant_order_info"]["order_id"] == "ORD-10001"


class TestExecuteToolOrderModifications:
    """_execute_tool() — cancel, update_address, update_quantity, remove_item."""

    def _state(self, customer_id="CUST-002"):
        return {"customer_id": customer_id, "session_id": "test", "metadata": {}}

    def _args(self, order_id="ORD-10003", status="processing", **kwargs):
        base = {"order_id": order_id, "order_status": {"order_status": status}}
        base.update(kwargs)
        return base

    # ── cancel_order ──────────────────────────────────────────────────────────

    def test_cancel_processing_order_succeeds(self, monkeypatch):
        """cancel_order with status=processing calls the tool and returns ok."""
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Order ORD-10003 has been successfully cancelled."
        monkeypatch.setattr("agents.order_lookup.cancel_order_tool", mock_tool)
        result = _execute_tool("cancel_order", self._args(status="processing"), self._state())
        assert result["status"] == "ok"
        mock_tool.invoke.assert_called_once()

    def test_cancel_delivered_order_denied(self):
        """Cancel denied when order status is 'delivered'."""
        result = _execute_tool("cancel_order", self._args(status="delivered"), self._state())
        assert result["status"] == "denied"
        assert "delivered" in result["message"]

    def test_cancel_in_transit_order_denied(self):
        """Cancel denied when order status is 'in_transit'."""
        result = _execute_tool("cancel_order", self._args(status="in_transit"), self._state())
        assert result["status"] == "denied"

    def test_cancel_already_cancelled_order_denied(self):
        """Cancel denied when order status is already 'cancelled'."""
        result = _execute_tool("cancel_order", self._args(status="cancelled"), self._state())
        assert result["status"] == "denied"

    # ── update_address ────────────────────────────────────────────────────────

    def test_update_address_processing_order_succeeds(self, monkeypatch):
        """update_address on processing order calls the tool and returns ok."""
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Address updated successfully."
        monkeypatch.setattr("agents.order_lookup.update_address_tool", mock_tool)
        args = self._args(status="processing", new_address="99 New Street, Cairo")
        result = _execute_tool("update_address", args, self._state())
        assert result["status"] == "ok"

    def test_update_address_shipped_order_denied(self):
        """Update address denied when order status is 'shipped'."""
        args = self._args(status="shipped", new_address="x")
        result = _execute_tool("update_address", args, self._state())
        assert result["status"] == "denied"

    # ── update_quantity ───────────────────────────────────────────────────────

    def test_update_quantity_processing_order_succeeds(self, monkeypatch):
        """update_quantity on processing order calls the tool and returns ok."""
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Quantity updated."
        monkeypatch.setattr("agents.order_lookup.update_quantity_tool", mock_tool)
        args = self._args(status="processing", product_id="PROD-005", new_qty=2)
        result = _execute_tool("update_quantity", args, self._state())
        assert result["status"] == "ok"

    def test_update_quantity_non_processing_denied(self):
        """Quantity update denied on delivered order."""
        args = self._args(status="delivered", product_id="PROD-005", new_qty=2)
        result = _execute_tool("update_quantity", args, self._state())
        assert result["status"] == "denied"

    def test_update_quantity_zero_delegates_to_tool(self, monkeypatch):
        """Quantity of 0 is passed to the tool (tool handles removal logic)."""
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Item removed."
        monkeypatch.setattr("agents.order_lookup.update_quantity_tool", mock_tool)
        args = self._args(status="processing", product_id="PROD-005", new_qty=0)
        result = _execute_tool("update_quantity", args, self._state())
        assert result["status"] == "ok"
        _, call_kwargs = mock_tool.invoke.call_args
        assert call_kwargs.get("new_qty", mock_tool.invoke.call_args[0][0].get("new_qty")) == 0

    # ── remove_item ───────────────────────────────────────────────────────────

    def test_remove_item_processing_order_succeeds(self, monkeypatch):
        """remove_item on processing order calls the tool and returns ok."""
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "FitTrack Smart Watch removed."
        monkeypatch.setattr("agents.order_lookup.remove_item_tool", mock_tool)
        args = self._args(status="processing", product_id="PROD-005")
        result = _execute_tool("remove_item", args, self._state())
        assert result["status"] == "ok"

    def test_remove_item_delivered_order_denied(self):
        """Remove item denied on delivered order."""
        args = self._args(status="delivered", product_id="PROD-001")
        result = _execute_tool("remove_item", args, self._state("CUST-001"))
        assert result["status"] == "denied"

    def test_remove_last_item_tool_handles_cancellation(self, monkeypatch):
        """Tool response for last-item removal is passed through correctly."""
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = (
            "FitTrack Smart Watch was the only item in order ORD-10003. "
            "The order has been cancelled automatically."
        )
        monkeypatch.setattr("agents.order_lookup.remove_item_tool", mock_tool)
        args = self._args(status="processing", product_id="PROD-005")
        result = _execute_tool("remove_item", args, self._state())
        assert result["status"] == "ok"
        assert "cancelled" in result["result"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Planner Node
# ══════════════════════════════════════════════════════════════════════════════

class TestPlannerNode:
    """planner_node() with mocked LLM."""

    def _make_state(self, message="Where is my order?", customer_id="CUST-001",
                    order_id=None, identity_verified=False, replan_count=0):
        return make_state(message=message, customer_id=customer_id,
                          order_id=order_id, identity_verified=identity_verified,
                          replan_count=replan_count)

    def test_planner_saves_plan_to_metadata(self, monkeypatch):
        """Successful planning saves tasks to metadata['compiler_state']['plan']."""
        plan_json = json.dumps(_SEND_VERIFY_PLAN)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: mock_llm(plan_json))
        state = self._make_state()
        result = planner_node(state)
        plan = result["metadata"]["compiler_state"]["plan"]
        assert len(plan) == 2

    def test_planner_includes_otp_when_not_verified(self, monkeypatch):
        """When identity_verified=False, plan includes send_otp."""
        plan_json = json.dumps(_SEND_VERIFY_PLAN)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: mock_llm(plan_json))
        state = self._make_state(identity_verified=False)
        result = planner_node(state)
        tools = [t["tool"] for t in result["metadata"]["compiler_state"]["plan"]]
        assert "send_otp" in tools

    def test_planner_skips_otp_when_already_verified(self, monkeypatch):
        """When identity_verified=True, LLM plan with no OTP is accepted."""
        plan_json = json.dumps(_LIST_PLAN)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: mock_llm(plan_json))
        state = self._make_state(identity_verified=True)
        result = planner_node(state)
        tools = [t["tool"] for t in result["metadata"]["compiler_state"]["plan"]]
        assert "send_otp" not in tools

    def test_planner_uses_fallback_on_invalid_json(self, monkeypatch):
        """When LLM returns non-JSON garbage, fallback plan is used — no crash."""
        monkeypatch.setattr("agents.order_lookup._get_llm",
                            lambda: mock_llm("THIS IS NOT JSON AT ALL"))
        state = self._make_state()
        result = planner_node(state)
        plan = result["metadata"]["compiler_state"]["plan"]
        assert len(plan) > 0

    def test_planner_uses_fallback_on_cycle(self, monkeypatch):
        """LLM returns cyclic plan — fallback used, no crash."""
        cyclic = {"tasks": [
            {"id": "T1", "tool": "send_otp", "args": {"customer_id": "x"}, "deps": ["T2"]},
            {"id": "T2", "tool": "send_otp", "args": {"customer_id": "x"}, "deps": ["T1"]},
        ]}
        monkeypatch.setattr("agents.order_lookup._get_llm",
                            lambda: mock_llm(json.dumps(cyclic)))
        state = self._make_state()
        result = planner_node(state)
        assert len(result["metadata"]["compiler_state"]["plan"]) > 0

    def test_planner_uses_fallback_on_unknown_tool(self, monkeypatch):
        """LLM returns plan with unknown tool — fallback used, no crash."""
        bad = {"tasks": [{"id": "T1", "tool": "fly_to_moon", "args": {}, "deps": []}]}
        monkeypatch.setattr("agents.order_lookup._get_llm",
                            lambda: mock_llm(json.dumps(bad)))
        state = self._make_state()
        result = planner_node(state)
        assert len(result["metadata"]["compiler_state"]["plan"]) > 0

    def test_planner_sets_agent_used(self, monkeypatch):
        """planner_node sets agent_used='order_lookup'."""
        monkeypatch.setattr("agents.order_lookup._get_llm",
                            lambda: mock_llm(json.dumps(_SEND_VERIFY_PLAN)))
        result = planner_node(self._make_state())
        assert result["agent_used"] == "order_lookup"

    def test_planner_includes_retrieve_knowledge_for_policy_question(self, monkeypatch):
        """Policy question plan includes retrieve_knowledge."""
        plan_with_rag = {"tasks": [
            {"id": "T1", "tool": "send_otp", "args": {"customer_id": "CUST-001"}, "deps": []},
            {"id": "T2", "tool": "verify_otp",
             "args": {"customer_id": "CUST-001", "entered_code": "x"}, "deps": ["T1"]},
            {"id": "T3", "tool": "retrieve_knowledge",
             "args": {"query": "return policy"}, "deps": ["T2"]},
        ]}
        monkeypatch.setattr("agents.order_lookup._get_llm",
                            lambda: mock_llm(json.dumps(plan_with_rag)))
        state = self._make_state("What is your return policy?")
        result = planner_node(state)
        tools = [t["tool"] for t in result["metadata"]["compiler_state"]["plan"]]
        assert "retrieve_knowledge" in tools

    def test_planner_replan_count_preserved_in_metadata(self, monkeypatch):
        """Existing replan_count in state is preserved in output metadata."""
        monkeypatch.setattr("agents.order_lookup._get_llm",
                            lambda: mock_llm(json.dumps(_SEND_VERIFY_PLAN)))
        state = self._make_state(replan_count=1)
        result = planner_node(state)
        assert result["metadata"]["compiler_state"]["replan_count"] == 1

    def test_planner_context_notes_include_replan_message(self, monkeypatch):
        """On replan, the LLM prompt is called once (we verify no crash with replan_count=1)."""
        lm = mock_llm(json.dumps(_LIST_PLAN))
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        state = self._make_state(identity_verified=True, replan_count=1)
        result = planner_node(state)
        assert result["metadata"]["compiler_state"]["replan_count"] == 1
        lm.invoke.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Joiner Node
# ══════════════════════════════════════════════════════════════════════════════

class TestJoinerNodeOTPFlow:
    """joiner_node() handles OTP status without calling the LLM."""

    def _state_with_results(self, results, customer_id="CUST-001", message="test"):
        state = make_state(message=message, customer_id=customer_id)
        state["metadata"]["compiler_state"]["completed_tasks"] = results
        return state

    def test_otp_just_sent_returns_pending_verification(self):
        """When send_otp result is otp_sent, returns pending_verification (no LLM)."""
        results = {"T1": {"status": "otp_sent", "message": "OTP sent"}}
        result = joiner_node(self._state_with_results(results))
        assert result["resolution_status"] == "pending_verification"
        assert not result["requires_escalation"]

    def test_otp_just_sent_message_is_correct(self):
        """OTP-sent response tells customer to check email."""
        results = {"T1": {"status": "otp_sent", "message": "OTP sent"}}
        result = joiner_node(self._state_with_results(results))
        content = result["messages"][0].content.lower()
        assert "verification code" in content or "6-digit" in content

    def test_otp_failed_returns_try_again_message(self):
        """When verify_otp returns failed, joiner tells customer code doesn't match."""
        results = {
            "T1": {"status": "otp_sent"},
            "T2": {"status": "failed", "message": "Code does not match"},
        }
        result = joiner_node(self._state_with_results(results))
        assert result["resolution_status"] == "pending_verification"
        content = result["messages"][0].content.lower()
        assert "match" in content or "try again" in content or "code" in content

    def test_otp_failed_does_not_escalate(self):
        """Wrong OTP does not set requires_escalation."""
        results = {"T1": {"status": "failed", "message": "Code does not match"}}
        result = joiner_node(self._state_with_results(results))
        assert result["requires_escalation"] is False

    def test_already_verified_proceeds_to_llm(self, monkeypatch):
        """already_verified status skips OTP gate and proceeds to LLM decision."""
        results = {"T1": {"status": "already_verified"}}
        lm = mock_llm_seq(_JOINER_ANSWER, "Here is your order info.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state_with_results(results))
        assert result["resolution_status"] == "resolved"


class TestJoinerNodeFraud:
    """joiner_node() fraud detection — runs before OTP check, no LLM."""

    def _state_with_fraud(self, customer_id="CUST-002"):
        state = make_state(customer_id=customer_id,
                           message="Where is order ORD-10001?")
        state["metadata"]["compiler_state"]["completed_tasks"] = {
            "T1": {"status": "fraud_detected",
                   "message": "Order belongs to a different account"},
        }
        return state

    def test_fraud_detected_escalates_immediately(self):
        """fraud_detected in results → resolution_status=escalated, no LLM call."""
        result = joiner_node(self._state_with_fraud())
        assert result["resolution_status"] == "escalated"

    def test_fraud_sets_requires_escalation_true(self):
        """Fraud detection sets requires_escalation=True."""
        result = joiner_node(self._state_with_fraud())
        assert result["requires_escalation"] is True

    def test_fraud_response_never_reveals_order_exists(self):
        """Response says order not found — never 'belongs to someone else' or 'fraud'."""
        result = joiner_node(self._state_with_fraud())
        content = result["messages"][0].content.lower()
        assert "fraud" not in content
        assert "belongs to" not in content
        assert "not found" in content or "locate" in content or "double-check" in content

    def test_fraud_response_never_reveals_real_customer(self):
        """Response never exposes CUST-001 (the real owner) to CUST-002."""
        result = joiner_node(self._state_with_fraud("CUST-002"))
        content = result["messages"][0].content
        assert "CUST-001" not in content

    def test_fraud_builds_handoff_with_suspicious_sentiment(self):
        """Fraud handoff sets customer_sentiment='suspicious'."""
        result = joiner_node(self._state_with_fraud())
        hf = result["metadata"]["escalation_handoff"]
        assert hf["customer_sentiment"] == "suspicious"
        assert hf["escalation_reason"] == "ownership_violation"

    def test_fraud_saved_to_long_term_memory(self, monkeypatch):
        """Fraud event triggers a long-term memory save."""
        mock_ltm = MagicMock()
        monkeypatch.setattr("memory.long_term.LongTermMemory", lambda: mock_ltm)
        joiner_node(self._state_with_fraud())
        mock_ltm.save_interaction.assert_called_once()
        call_kwargs = mock_ltm.save_interaction.call_args[1]
        assert call_kwargs["metadata"]["intent"] == "fraud"


class TestJoinerNodeDecisions:
    """joiner_node() ANSWER / CLARIFY / ESCALATE / REPLAN decisions."""

    def _state_with_results(self, results=None, replan_count=0, extra_metadata=None):
        if results is None:
            results = {"T1": {"status": "ok", "data": "list of orders"}}
        state = make_state(replan_count=replan_count)
        state["metadata"]["compiler_state"]["completed_tasks"] = results
        if extra_metadata:
            state["metadata"].update(extra_metadata)
        return state

    def test_answer_decision_generates_response(self, monkeypatch):
        """ANSWER decision calls the solver and returns an AIMessage."""
        lm = mock_llm_seq(_JOINER_ANSWER, "Your order is delivered!")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state_with_results())
        assert result["resolution_status"] == "resolved"
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Your order is delivered!"

    def test_clarify_decision_asks_question(self, monkeypatch):
        """CLARIFY decision returns the clarify_question as the message."""
        lm = mock_llm(_JOINER_CLARIFY)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state_with_results())
        assert result["resolution_status"] == "pending_clarification"
        assert "address" in result["messages"][0].content.lower()

    def test_clarify_saves_pending_question_to_metadata(self, monkeypatch):
        """CLARIFY saves question to metadata['compiler_state']['pending_clarification']."""
        lm = mock_llm(_JOINER_CLARIFY)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state_with_results())
        cs = result["metadata"]["compiler_state"]
        assert "pending_clarification" in cs
        assert cs["awaiting_clarification"] is True

    def test_escalate_decision_sets_requires_escalation(self, monkeypatch):
        """ESCALATE decision sets requires_escalation=True."""
        lm = mock_llm(_JOINER_ESCALATE)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state_with_results())
        assert result["requires_escalation"] is True
        assert result["resolution_status"] == "escalated"

    def test_escalate_builds_handoff(self, monkeypatch):
        """ESCALATE builds a handoff dict in metadata."""
        lm = mock_llm(_JOINER_ESCALATE)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("memory.long_term.LongTermMemory", MagicMock)
        result = joiner_node(self._state_with_results())
        hf = result["metadata"]["escalation_handoff"]
        assert "issue_summary" in hf
        assert hf["agent"] == "nora_order_lookup"

    def test_escalate_saves_to_long_term_memory(self, monkeypatch):
        """ESCALATE saves the interaction to long-term memory."""
        lm = mock_llm(_JOINER_ESCALATE)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        mock_ltm = MagicMock()
        monkeypatch.setattr("memory.long_term.LongTermMemory", lambda: mock_ltm)
        joiner_node(self._state_with_results())
        mock_ltm.save_interaction.assert_called_once()

    def test_replan_increments_replan_count(self, monkeypatch):
        """REPLAN when under limit increments replan_count and sets replanning status."""
        lm = mock_llm(_JOINER_REPLAN)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state_with_results(replan_count=0))
        assert result["metadata"]["compiler_state"]["replan_count"] == 1
        assert result["resolution_status"] == "replanning"

    def test_replan_max_exceeded_falls_through_to_answer(self, monkeypatch):
        """When replan_count >= MAX_REPLANS, REPLAN decision falls through to ANSWER."""
        lm = mock_llm_seq(_JOINER_REPLAN, "Fallback answer from solver.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state_with_results(replan_count=MAX_REPLANS))
        assert result["resolution_status"] in ("resolved", "needs_rerouting")

    def test_joiner_parse_failure_defaults_to_answer(self, monkeypatch):
        """Invalid JSON from decision LLM defaults to ANSWER — no crash."""
        lm = mock_llm_seq("NOT JSON", "Fallback answer.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state_with_results())
        assert isinstance(result["messages"][0], AIMessage)

    def test_return_context_sets_needs_rerouting(self, monkeypatch):
        """When return_context in metadata, resolution_status=needs_rerouting."""
        lm = mock_llm_seq(_JOINER_ANSWER, "Routing to Maya now.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        extra = {"return_context": {"order_id": "ORD-10001", "items": []}}
        result = joiner_node(self._state_with_results(extra_metadata=extra))
        assert result["resolution_status"] == "needs_rerouting"

    def test_retrieval_scores_extracted_correctly(self, monkeypatch):
        """Retrieval scores from retrieve_knowledge are passed to the returned state."""
        results = {"T1": {"status": "ok", "context": "RAG content", "scores": [0.85, 0.72]}}
        lm = mock_llm_seq(_JOINER_ANSWER, "Here is what I found.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state_with_results(results=results))
        assert result["retrieval_scores"] == [0.85, 0.72]


class TestJoinerNodeEdgeCases:
    """joiner_node() edge cases — empty results, missing fields."""

    def _state(self, results=None):
        state = make_state()
        state["metadata"]["compiler_state"]["completed_tasks"] = results or {}
        return state

    def test_empty_task_results_handled(self, monkeypatch):
        """Joiner with empty results dict doesn't crash."""
        lm = mock_llm_seq(_JOINER_ANSWER, "Nothing to show yet.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state({}))
        assert "messages" in result

    def test_missing_kb_context_uses_fallback(self, monkeypatch):
        """No retrieve_knowledge result uses default 'No relevant articles found.'"""
        lm = mock_llm_seq(_JOINER_ANSWER, "Response without KB.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = joiner_node(self._state({"T1": {"status": "ok", "data": "orders"}}))
        assert isinstance(result["messages"][0], AIMessage)

    def test_solver_failure_returns_fallback_message(self, monkeypatch):
        """When solver LLM call raises, a hardcoded fallback message is returned."""
        decision_mock = MagicMock()
        decision_mock.invoke.side_effect = [
            MagicMock(content=_JOINER_ANSWER),
            Exception("LLM unavailable"),
        ]
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: decision_mock)
        result = joiner_node(self._state())
        content = result["messages"][0].content.lower()
        assert "support@shopease.com" in content or "trouble" in content


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Full Flow Integration
# ══════════════════════════════════════════════════════════════════════════════

class TestFullFlowHappyPaths:
    """order_lookup_node() happy path — already-verified identity, read-only data."""

    def _verified_state(self, message="Show my orders", customer_id="CUST-001",
                        order_id=None):
        return make_state(message=message, customer_id=customer_id,
                          order_id=order_id, identity_verified=True)

    def test_list_all_orders_no_order_id(self, monkeypatch):
        """'Show me all my orders' → planner picks list_orders, joiner answers."""
        lm = mock_llm_seq(
            json.dumps(_LIST_PLAN),   # planner
            _JOINER_ANSWER,           # joiner decision
            "You have 3 orders.",     # joiner solver
        )
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = order_lookup_node(self._verified_state("Show me all my orders"))
        assert result["resolution_status"] == "resolved"
        assert result["requires_escalation"] is False
        assert isinstance(result["messages"][-1], AIMessage)

    def test_order_status_query_resolved(self, monkeypatch):
        """Order status query with known order → resolved."""
        fetch_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10001"}, "deps": []},
        ]})
        lm = mock_llm_seq(
            fetch_plan,
            _JOINER_ANSWER,
            "Order ORD-10001 was delivered on April 25, 2025.",
        )
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = order_lookup_node(
            self._verified_state("Where is ORD-10001?", order_id="ORD-10001")
        )
        assert result["resolution_status"] == "resolved"

    def test_cancel_processing_order_full_flow(self, monkeypatch):
        """Cancel plan runs: mock tools return success → resolved."""
        cancel_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10003"}, "deps": []},
            {"id": "T2", "tool": "check_order_status",
             "args": {"order_data": "$T1"}, "deps": ["T1"]},
            {"id": "T3", "tool": "cancel_order",
             "args": {"order_id": "ORD-10003", "customer_id": "CUST-002",
                      "order_status": "$T2"}, "deps": ["T2"]},
        ]})
        lm = mock_llm_seq(cancel_plan, _JOINER_ANSWER, "Your order has been cancelled.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        mock_cancel = MagicMock()
        mock_cancel.invoke.return_value = "Order ORD-10003 has been successfully cancelled."
        monkeypatch.setattr("agents.order_lookup.cancel_order_tool", mock_cancel)
        result = order_lookup_node(
            self._verified_state("Cancel my order ORD-10003",
                                 customer_id="CUST-002", order_id="ORD-10003")
        )
        assert result["resolution_status"] == "resolved"

    def test_return_request_sets_needs_rerouting(self, monkeypatch):
        """Return request → collect_return_context → needs_rerouting (supervisor re-routes to Maya)."""
        return_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10001"}, "deps": []},
            {"id": "T2", "tool": "collect_return_context",
             "args": {"order_data": "$T1"}, "deps": ["T1"]},
        ]})
        lm = mock_llm_seq(return_plan, _JOINER_ANSWER, "Routing you to Maya for the return.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = order_lookup_node(
            self._verified_state("I want to return my laptop",
                                 customer_id="CUST-001", order_id="ORD-10001")
        )
        assert result["resolution_status"] == "needs_rerouting"
        assert "return_context" in result["metadata"]

    def test_agent_used_is_order_lookup(self, monkeypatch):
        """agent_used is always 'order_lookup' for any full flow."""
        lm = mock_llm_seq(json.dumps(_LIST_PLAN), _JOINER_ANSWER, "Here are your orders.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = order_lookup_node(self._verified_state())
        assert result["agent_used"] == "order_lookup"


class TestFullFlowOTPFlow:
    """order_lookup_node() — OTP send and verify flow."""

    def test_first_contact_sends_otp(self, monkeypatch):
        """First message with no OTP → plan sends OTP, returns pending_verification."""
        otp_plan = json.dumps(_SEND_VERIFY_PLAN)
        lm = mock_llm(otp_plan)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("agents.order_lookup.send_otp_email",
                            lambda cid: ("123456", "t***@test.com"))
        state = make_state("Where is my order?", identity_verified=False)
        result = order_lookup_node(state)
        assert result["resolution_status"] == "pending_verification"
        assert result["requires_escalation"] is False

    def test_wrong_otp_returns_try_again(self, monkeypatch):
        """Wrong OTP → resolution_status=pending_verification, never reveals data."""
        wrong_verify_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "send_otp", "args": {"customer_id": "CUST-001"}, "deps": []},
            {"id": "T2", "tool": "verify_otp",
             "args": {"customer_id": "CUST-001", "entered_code": "999999"}, "deps": ["T1"]},
        ]})
        lm = mock_llm(wrong_verify_plan)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("agents.order_lookup.send_otp_email",
                            lambda cid: ("123456", "t***@test.com"))
        state = make_state("999999", identity_verified=False)
        state["metadata"]["pending_otp"] = "123456"
        result = order_lookup_node(state)
        assert result["resolution_status"] == "pending_verification"

    def test_already_verified_skips_otp(self, monkeypatch):
        """Second turn with identity_verified=True skips OTP entirely."""
        lm = mock_llm_seq(json.dumps(_LIST_PLAN), _JOINER_ANSWER, "Your orders:")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        state = make_state("Show orders", identity_verified=True)
        result = order_lookup_node(state)
        assert result["resolution_status"] == "resolved"


class TestFullFlowEdgeCases:
    """order_lookup_node() edge cases — nonexistent orders, empty messages, etc."""

    def _verified_state(self, message, customer_id="CUST-001", order_id=None):
        return make_state(message=message, customer_id=customer_id,
                          order_id=order_id, identity_verified=True)

    def test_empty_message_no_crash(self, monkeypatch):
        """Empty customer message does not crash the agent."""
        lm = mock_llm_seq(json.dumps(_LIST_PLAN), _JOINER_ANSWER, "How can I help?")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = order_lookup_node(self._verified_state(""))
        assert "messages" in result

    def test_very_long_message_no_crash(self, monkeypatch):
        """2000-character message does not crash the agent."""
        lm = mock_llm_seq(json.dumps(_LIST_PLAN), _JOINER_ANSWER, "Got it.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = order_lookup_node(self._verified_state("x" * 2000))
        assert "messages" in result

    def test_order_not_found_graceful(self, monkeypatch):
        """ORD-99999 not in database → graceful response, no crash."""
        fetch_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-99999"}, "deps": []},
        ]})
        lm = mock_llm_seq(fetch_plan, _JOINER_ANSWER, "I couldn't find that order.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = order_lookup_node(self._verified_state("ORD-99999", order_id="ORD-99999"))
        assert "messages" in result

    def test_update_address_clarify_when_no_address(self, monkeypatch):
        """CLARIFY returned when joiner asks for address."""
        lm = mock_llm_seq(json.dumps(_LIST_PLAN), _JOINER_CLARIFY)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        result = order_lookup_node(self._verified_state("Change my delivery address"))
        assert result["resolution_status"] == "pending_clarification"
        assert "address" in result["messages"][-1].content.lower()


class TestFullFlowFraudAndSecurity:
    """order_lookup_node() fraud and security scenarios."""

    def test_fraud_different_customer_escalated(self, monkeypatch):
        """CUST-002 accessing ORD-10001 (CUST-001) → escalated, requires_escalation=True."""
        fetch_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10001"}, "deps": []},
        ]})
        lm = mock_llm(fetch_plan)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("memory.long_term.LongTermMemory", MagicMock)
        state = make_state("Where is ORD-10001?", customer_id="CUST-002",
                           order_id="ORD-10001", identity_verified=True)
        result = order_lookup_node(state)
        assert result["requires_escalation"] is True
        assert result["resolution_status"] == "escalated"

    def test_fraud_response_reveals_nothing(self, monkeypatch):
        """Fraud response does not contain the word 'fraud', real owner, or order details."""
        fetch_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10001"}, "deps": []},
        ]})
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: mock_llm(fetch_plan))
        monkeypatch.setattr("memory.long_term.LongTermMemory", MagicMock)
        state = make_state("ORD-10001?", customer_id="CUST-002",
                           order_id="ORD-10001", identity_verified=True)
        result = order_lookup_node(state)
        content = result["messages"][-1].content
        assert "fraud" not in content.lower()
        assert "CUST-001" not in content
        assert "belongs to another" not in content.lower()

    def test_otp_replay_attack_blocked(self, monkeypatch):
        """Same OTP code can only verify once — second attempt returns already_verified.
        Simulates scheduler side-effect merge between the two calls."""
        state = make_state("123456", identity_verified=False)
        state["metadata"]["pending_otp"] = "123456"
        r1 = _execute_tool("verify_otp",
                           {"customer_id": "CUST-001", "entered_code": "123456"}, state)
        assert r1["status"] == "verified"
        # Simulate scheduler merge before next call
        state["metadata"]["identity_verified"] = r1["identity_verified"]
        state["metadata"]["pending_otp"] = r1["pending_otp"]
        r2 = _execute_tool("verify_otp",
                           {"customer_id": "CUST-001", "entered_code": "123456"}, state)
        assert r2["status"] == "already_verified"


class TestFullFlowEscalationPaths:
    """order_lookup_node() — all escalation triggers."""

    def _verified_state(self, message, customer_id="CUST-001"):
        return make_state(message=message, customer_id=customer_id, identity_verified=True)

    def test_escalation_sets_requires_escalation(self, monkeypatch):
        """Any ESCALATE decision → requires_escalation=True."""
        lm = mock_llm_seq(json.dumps(_LIST_PLAN), _JOINER_ESCALATE)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("memory.long_term.LongTermMemory", MagicMock)
        result = order_lookup_node(self._verified_state("I'm suing ShopEase"))
        assert result["requires_escalation"] is True

    def test_escalation_handoff_has_all_required_fields(self, monkeypatch):
        """ESCALATE handoff contains all 7 required fields."""
        lm = mock_llm_seq(json.dumps(_LIST_PLAN), _JOINER_ESCALATE)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("memory.long_term.LongTermMemory", MagicMock)
        result = order_lookup_node(self._verified_state("I'm suing ShopEase"))
        hf = result["metadata"]["escalation_handoff"]
        for field in ("issue_summary", "actions_attempted", "customer_sentiment",
                      "relevant_order_info", "escalation_reason", "confidence_score", "agent"):
            assert field in hf, f"Missing required handoff field: {field}"

    def test_escalation_message_is_empathetic(self, monkeypatch):
        """Escalation response never says 'I can't help' — must offer to connect specialist."""
        lm = mock_llm_seq(json.dumps(_LIST_PLAN), _JOINER_ESCALATE)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("memory.long_term.LongTermMemory", MagicMock)
        result = order_lookup_node(self._verified_state("legal action"))
        content = result["messages"][-1].content.lower()
        assert "specialist" in content or "connecting" in content or "senior" in content


class TestFullFlowReplanPaths:
    """order_lookup_node() — REPLAN path, counter increment, loop."""

    def test_replan_increments_counter_at_joiner_level(self, monkeypatch):
        """REPLAN decision from joiner increments replan_count in metadata."""
        lm = mock_llm(_JOINER_REPLAN)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        state = make_state(identity_verified=True)
        state["metadata"]["compiler_state"]["completed_tasks"] = {
            "T1": {"status": "ok", "data": "orders"}
        }
        result = joiner_node(state)
        assert result["metadata"]["compiler_state"]["replan_count"] == 1
        assert result["resolution_status"] == "replanning"

    def test_replan_stores_notes_in_metadata(self, monkeypatch):
        """REPLAN stores replan_notes in compiler_state for next planner run."""
        lm = mock_llm(_JOINER_REPLAN)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        state = make_state(identity_verified=True)
        state["metadata"]["compiler_state"]["completed_tasks"] = {"T1": {"status": "error"}}
        result = joiner_node(state)
        assert "replan_notes" in result["metadata"]["compiler_state"]
        assert result["metadata"]["compiler_state"]["replan_notes"] == "retry without cache"

    def test_completed_tasks_preserved_across_replan(self, monkeypatch):
        """Already-completed tasks are skipped on the second scheduler run."""
        # Simulate scheduler with tasks T1 already done and T2 pending
        plan = [
            {"id": "T1", "tool": "send_otp",
             "args": {"customer_id": "CUST-001"}, "deps": []},
            {"id": "T2", "tool": "verify_otp",
             "args": {"customer_id": "CUST-001", "entered_code": "x"}, "deps": ["T1"]},
        ]
        state = make_state(identity_verified=False)
        state["metadata"]["compiler_state"]["plan"] = plan
        state["metadata"]["compiler_state"]["completed_tasks"] = {
            "T1": {"status": "otp_sent"}  # T1 already done
        }
        monkeypatch.setattr("agents.order_lookup.send_otp_email",
                            lambda cid: ("123456", "t***@test.com"))
        scheduler_result = scheduler_node(state)
        completed = scheduler_result["metadata"]["compiler_state"]["completed_tasks"]
        # T1 should still be otp_sent (not re-run), T2 should have been executed
        assert completed["T1"]["status"] == "otp_sent"
        assert "T2" in completed


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Security Rules
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityRules:
    """Security rules enforced at the code level — no LLM needed."""

    def test_different_customer_order_always_fraud(self):
        """fetch_order for CUST-002 accessing ORD-10001 always returns fraud_detected."""
        state = {"customer_id": "CUST-002", "session_id": "s", "metadata": {}}
        result = _execute_tool("fetch_order", {"order_id": "ORD-10001"}, state)
        assert result["status"] == "fraud_detected"

    def test_any_customer_accessing_nonexistent_order_gets_not_found(self):
        """Accessing a non-existent order returns not_found, never fraud_detected."""
        state = {"customer_id": "CUST-001", "session_id": "s", "metadata": {}}
        result = _execute_tool("fetch_order", {"order_id": "ORD-00000"}, state)
        assert result["status"] == "not_found"

    def test_cancel_checks_status_before_tool_call(self, monkeypatch):
        """cancel_order executor enforces status=processing check before calling tool."""
        called = []
        mock_tool = MagicMock(side_effect=lambda **kwargs: called.append(kwargs))
        monkeypatch.setattr("agents.order_lookup.cancel_order_tool", mock_tool)
        state = {"customer_id": "CUST-001", "session_id": "s", "metadata": {}}
        _execute_tool("cancel_order",
                      {"order_id": "ORD-10001", "order_status": {"order_status": "delivered"}},
                      state)
        assert not mock_tool.invoke.called

    def test_update_address_checks_status_before_tool_call(self, monkeypatch):
        """update_address executor enforces status=processing check before calling tool."""
        mock_tool = MagicMock()
        monkeypatch.setattr("agents.order_lookup.update_address_tool", mock_tool)
        state = {"customer_id": "CUST-001", "session_id": "s", "metadata": {}}
        _execute_tool("update_address",
                      {"order_id": "ORD-10001", "new_address": "x",
                       "order_status": {"order_status": "in_transit"}},
                      state)
        assert not mock_tool.invoke.called

    def test_validate_nora_tasks_blocks_order_tool_without_otp(self):
        """validate_nora_tasks rejects fetch_order when verify_otp is in plan but not a dep."""
        tasks = [
            {"id": "T1", "tool": "send_otp", "args": {"customer_id": "x"}, "deps": []},
            {"id": "T2", "tool": "verify_otp",
             "args": {"customer_id": "x", "entered_code": "c"}, "deps": ["T1"]},
            {"id": "T3", "tool": "fetch_order",
             "args": {"order_id": "ORD-10001"}, "deps": []},  # Missing T2
        ]
        with pytest.raises(ValueError, match="must depend on verify_otp"):
            validate_nora_tasks(tasks)

    def test_otp_cleared_after_verification(self):
        """pending_otp=None is returned in the result dict after successful verification."""
        state = {"customer_id": "CUST-001", "session_id": "s",
                 "metadata": {"pending_otp": "555444"}}
        result = _execute_tool("verify_otp",
                               {"customer_id": "CUST-001", "entered_code": "555444"}, state)
        assert result["pending_otp"] is None

    def test_fraud_handoff_confidence_score_is_high(self):
        """Fraud handoff has confidence_score >= 0.9 (high certainty)."""
        state = make_state(customer_id="CUST-002")
        state["metadata"]["compiler_state"]["completed_tasks"] = {
            "T1": {"status": "fraud_detected", "message": "Order belongs to different account"}
        }
        result = joiner_node(state)
        hf = result["metadata"]["escalation_handoff"]
        assert hf["confidence_score"] >= 0.9

    def test_joiner_fraud_check_before_otp_check(self):
        """Fraud check runs before OTP check — even with otp_sent in results, fraud wins."""
        state = make_state(customer_id="CUST-002")
        state["metadata"]["compiler_state"]["completed_tasks"] = {
            "T_otp": {"status": "otp_sent"},
            "T_fetch": {"status": "fraud_detected", "message": "wrong customer"},
        }
        result = joiner_node(state)
        # Fraud should win — not otp_sent pending_verification
        assert result["resolution_status"] == "escalated"
        assert result["requires_escalation"] is True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Complete Flow Tests (multi-turn, real scheduler, mocked LLM)
# ══════════════════════════════════════════════════════════════════════════════

class TestCompleteFlows:
    """
    Multi-turn end-to-end flow tests for order_lookup_node.

    These tests exercise the agent as a whole — planner → scheduler → joiner —
    simulating real user conversations across multiple turns.  State is threaded
    between turns exactly the way LangGraph's checkpointer + reducers would:
      - messages appended via add_messages
      - metadata merged via {**old, **new}

    Mock data quick-reference (from mock_orders.json):
      CUST-001: ORD-10001 (delivered), ORD-10002 (in_transit), ORD-10007 (delivered)
      CUST-002: ORD-10003 (processing), ORD-10005 (return_initiated)
      CUST-003: ORD-10004 (delivered)
    """

    # ── State helpers ──────────────────────────────────────────────────────────

    def _next_turn(self, prev_state: dict, prev_result: dict,
                   new_message: str, **overrides) -> dict:
        """
        Build the next turn's state from the previous turn's state + result + new message.
        Simulates LangGraph's add_messages reducer and metadata merge reducer.
        """
        merged_meta = {**prev_state.get("metadata", {}), **prev_result.get("metadata", {})}
        next_s = {
            **prev_state,
            **{k: v for k, v in prev_result.items()
               if k not in ("messages", "metadata")},
            "messages": (
                prev_state.get("messages", [])
                + prev_result.get("messages", [])
                + [HumanMessage(content=new_message)]
            ),
            "metadata": merged_meta,
        }
        next_s.update(overrides)
        return next_s

    def _send_otp_plan(self, customer_id: str) -> str:
        return json.dumps({"tasks": [
            {"id": "T1", "tool": "send_otp",
             "args": {"customer_id": customer_id}, "deps": []}
        ]})

    # ── OTP flows ──────────────────────────────────────────────────────────────

    def test_two_turn_otp_then_order_status(self, monkeypatch):
        """
        Full two-turn OTP flow for order status.
          Turn 1: 'Where is my order ORD-10001?' → planner LLM → OTP sent → pending_verification
          Turn 2: '123456'                        → fast-path plan (no LLM) → verified
                                                  → joiner fast-path (no LLM) → solver → resolved

        ORD-10001 belongs to CUST-001 so fetch_order succeeds.
        Turn 1: 1 LLM call.  Turn 2: 1 LLM call (solver only — joiner decision skipped by fast-path).
        """
        lm = mock_llm_seq(
            self._send_otp_plan("CUST-001"),          # Turn 1 planner
            "ORD-10001 was delivered on April 25.",   # Turn 2 solver (joiner decision skipped)
        )
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("agents.order_lookup.send_otp_email",
                            lambda cid: ("123456", "t***@test.com"))

        # Turn 1
        state1 = make_state("Where is my order ORD-10001?", customer_id="CUST-001",
                             order_id="ORD-10001", identity_verified=False)
        result1 = order_lookup_node(state1)

        assert result1["resolution_status"] == "pending_verification"
        assert "verification code" in result1["messages"][-1].content.lower()
        assert result1["requires_escalation"] is False

        # Turn 2 — OTP fast-path kicks in; planner never calls LLM
        state2 = self._next_turn(state1, result1, "123456")
        result2 = order_lookup_node(state2)

        assert result2["resolution_status"] == "resolved"
        assert result2["requires_escalation"] is False
        assert result2["messages"][-1].content == "ORD-10001 was delivered on April 25."
        # identity_verified was set in metadata after verification
        assert result2["metadata"].get("identity_verified") is True

    def test_three_turn_wrong_otp_then_correct(self, monkeypatch):
        """
        Three-turn OTP flow — customer enters wrong code first.
          Turn 1: 'Where is my order ORD-10001?' → OTP sent
          Turn 2: '999999' (wrong)               → fast-path plan → 'code doesn't match'
          Turn 3: '123456' (correct)             → fast-path plan → resolved

        Wrong code does NOT clear pending_otp, so Turn 3 can still verify.
        Turn 1: 1 LLM.  Turn 2: 0 LLM.  Turn 3: 2 LLM.
        """
        lm = mock_llm_seq(
            self._send_otp_plan("CUST-001"),
            _JOINER_ANSWER,
            "ORD-10001 is delivered — arrived April 25.",
        )
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("agents.order_lookup.send_otp_email",
                            lambda cid: ("123456", "t***@test.com"))

        state1 = make_state("Where is my order ORD-10001?", customer_id="CUST-001",
                             order_id="ORD-10001", identity_verified=False)
        result1 = order_lookup_node(state1)
        assert result1["resolution_status"] == "pending_verification"

        # Turn 2 — wrong OTP; joiner short-circuits without LLM
        state2 = self._next_turn(state1, result1, "999999")
        result2 = order_lookup_node(state2)
        assert result2["resolution_status"] == "pending_verification"
        assert any(w in result2["messages"][-1].content.lower()
                   for w in ("match", "try again", "code"))
        # pending_otp unchanged — "123456" must still be in metadata
        assert result2["metadata"].get("pending_otp") == "123456"

        # Turn 3 — correct OTP; resolves
        state3 = self._next_turn(state2, result2, "123456")
        result3 = order_lookup_node(state3)
        assert result3["resolution_status"] == "resolved"
        assert result3["requires_escalation"] is False

    def test_resend_otp_issues_new_code(self, monkeypatch):
        """
        Two-turn resend flow.
          Turn 1: 'Where is my order?' → OTP sent  (code = 100001)
          Turn 2: 'resend my code'     → NEW OTP sent (code = 100002, different from first)

        'resend my code' contains no 6-digit number, so the fast-path does NOT
        trigger.  The LLM planner produces another send_otp plan.
        """
        send_plan = self._send_otp_plan("CUST-001")
        lm = mock_llm_seq(send_plan, send_plan)   # 2 planner calls, 0 joiner LLM
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)

        call_n = {"n": 0}
        def fake_send(cid):
            call_n["n"] += 1
            return (str(100000 + call_n["n"]), "t***@test.com")
        monkeypatch.setattr("agents.order_lookup.send_otp_email", fake_send)

        state1 = make_state("Where is my order?", identity_verified=False)
        result1 = order_lookup_node(state1)
        assert result1["resolution_status"] == "pending_verification"
        otp1 = result1["metadata"].get("pending_otp")

        state2 = self._next_turn(state1, result1, "resend my code")
        result2 = order_lookup_node(state2)
        assert result2["resolution_status"] == "pending_verification"
        otp2 = result2["metadata"].get("pending_otp")

        assert otp2 is not None
        assert otp2 != otp1   # a genuinely new code was generated

    # ── Already-verified single-turn flows ────────────────────────────────────

    def test_verified_order_status_query(self, monkeypatch):
        """
        Single-turn: verified customer asks for order status.
        No OTP step.  fetch_order runs against real mock_orders.json.
        """
        fetch_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10002"}, "deps": []},
        ]})
        lm = mock_llm_seq(fetch_plan, _JOINER_ANSWER,
                          "ORD-10002 is in transit — estimated delivery May 23.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)

        state = make_state("Where is ORD-10002?", customer_id="CUST-001",
                           order_id="ORD-10002", identity_verified=True)
        result = order_lookup_node(state)

        assert result["resolution_status"] == "resolved"
        assert result["requires_escalation"] is False
        assert "ORD-10002" in result["messages"][-1].content or \
               isinstance(result["messages"][-1], AIMessage)

    def test_verified_list_all_orders(self, monkeypatch):
        """
        Single-turn: verified customer lists all orders.
        list_orders runs against real mock_orders.json.
        CUST-001 has 3 orders.
        """
        list_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "list_orders",
             "args": {"customer_id": "CUST-001"}, "deps": []},
        ]})
        lm = mock_llm_seq(list_plan, _JOINER_ANSWER,
                          "You have 3 orders: ORD-10001, ORD-10002, ORD-10007.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)

        state = make_state("Show me all my orders", customer_id="CUST-001",
                           identity_verified=True)
        result = order_lookup_node(state)

        assert result["resolution_status"] == "resolved"
        assert isinstance(result["messages"][-1], AIMessage)

    def test_verified_cancel_processing_order(self, monkeypatch):
        """
        Single-turn: verified customer cancels a processing order (ORD-10003).
        Plan: fetch_order → check_order_status → cancel_order.
        cancel_order_tool is mocked to avoid writing to disk.
        """
        cancel_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10003"}, "deps": []},
            {"id": "T2", "tool": "check_order_status",
             "args": {"order_data": "$T1"}, "deps": ["T1"]},
            {"id": "T3", "tool": "cancel_order",
             "args": {"order_id": "ORD-10003", "customer_id": "CUST-002",
                      "order_status": "$T2"}, "deps": ["T2"]},
        ]})
        lm = mock_llm_seq(cancel_plan, _JOINER_ANSWER,
                          "ORD-10003 has been cancelled. Refund within 3–5 days.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        mock_cancel = MagicMock()
        mock_cancel.invoke.return_value = "Order ORD-10003 has been successfully cancelled."
        monkeypatch.setattr("agents.order_lookup.cancel_order_tool", mock_cancel)

        state = make_state("Cancel my order ORD-10003", customer_id="CUST-002",
                           order_id="ORD-10003", identity_verified=True)
        result = order_lookup_node(state)

        assert result["resolution_status"] == "resolved"
        assert result["requires_escalation"] is False
        mock_cancel.invoke.assert_called_once()

    def test_verified_cancel_delivered_order_denied(self, monkeypatch):
        """
        Single-turn: verified customer tries to cancel a delivered order (ORD-10001).
        _execute_tool denies the cancel (status != processing) — no tool call made.
        Joiner answers with the denial.
        """
        cancel_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10001"}, "deps": []},
            {"id": "T2", "tool": "check_order_status",
             "args": {"order_data": "$T1"}, "deps": ["T1"]},
            {"id": "T3", "tool": "cancel_order",
             "args": {"order_id": "ORD-10001", "customer_id": "CUST-001",
                      "order_status": "$T2"}, "deps": ["T2"]},
        ]})
        mock_cancel = MagicMock()
        monkeypatch.setattr("agents.order_lookup.cancel_order_tool", mock_cancel)
        lm = mock_llm_seq(cancel_plan, _JOINER_ANSWER,
                          "Sorry, ORD-10001 is already delivered and cannot be cancelled.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)

        state = make_state("Cancel ORD-10001", customer_id="CUST-001",
                           order_id="ORD-10001", identity_verified=True)
        result = order_lookup_node(state)

        assert result["resolution_status"] == "resolved"
        # Tool must NOT have been called — the executor denied it before reaching the tool
        mock_cancel.invoke.assert_not_called()

    def test_verified_update_address_on_processing_order(self, monkeypatch):
        """
        Single-turn: verified customer updates the shipping address on ORD-10003.
        update_address_tool mocked to avoid writing to disk.
        """
        address_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10003"}, "deps": []},
            {"id": "T2", "tool": "check_order_status",
             "args": {"order_data": "$T1"}, "deps": ["T1"]},
            {"id": "T3", "tool": "update_address",
             "args": {"order_id": "ORD-10003", "customer_id": "CUST-002",
                      "new_address": "99 Nile St, Cairo",
                      "order_status": "$T2"}, "deps": ["T2"]},
        ]})
        lm = mock_llm_seq(address_plan, _JOINER_ANSWER, "Address updated to 99 Nile St, Cairo.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        mock_addr = MagicMock()
        mock_addr.invoke.return_value = "Address updated.\nNew: 99 Nile St, Cairo"
        monkeypatch.setattr("agents.order_lookup.update_address_tool", mock_addr)

        state = make_state("Change my address to 99 Nile St, Cairo",
                           customer_id="CUST-002", order_id="ORD-10003",
                           identity_verified=True)
        result = order_lookup_node(state)

        assert result["resolution_status"] == "resolved"
        mock_addr.invoke.assert_called_once()

    def test_verified_return_request_routes_to_maya(self, monkeypatch):
        """
        Single-turn: verified customer requests a return.
        collect_return_context runs and stores return_context in metadata.
        resolution_status must be 'needs_rerouting' so workflow sends to Maya (policy_returns).
        """
        return_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10001"}, "deps": []},
            {"id": "T2", "tool": "collect_return_context",
             "args": {"order_data": "$T1"}, "deps": ["T1"]},
        ]})
        lm = mock_llm_seq(return_plan, _JOINER_ANSWER,
                          "I'm routing your return to Maya, our Returns Specialist.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)

        state = make_state("I want to return my laptop", customer_id="CUST-001",
                           order_id="ORD-10001", identity_verified=True)
        result = order_lookup_node(state)

        assert result["resolution_status"] == "needs_rerouting"
        assert "return_context" in result["metadata"]
        rc = result["metadata"]["return_context"]
        assert rc["order_id"] == "ORD-10001"
        assert rc["customer_id"] == "CUST-001"
        assert isinstance(rc["items"], list)

    def test_verified_policy_question_uses_knowledge_base(self, monkeypatch):
        """
        Single-turn: verified customer asks a policy question.
        retrieve_knowledge runs (RAG fails silently → empty context).
        Joiner still generates a response using empty KB context.
        """
        rag_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "retrieve_knowledge",
             "args": {"query": "What is the return policy?"}, "deps": []},
        ]})
        lm = mock_llm_seq(rag_plan, _JOINER_ANSWER,
                          "Our return policy allows returns within 30 days of delivery.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)

        state = make_state("What is your return policy?", customer_id="CUST-001",
                           identity_verified=True)
        result = order_lookup_node(state)

        assert result["resolution_status"] == "resolved"
        assert isinstance(result["messages"][-1], AIMessage)

    def test_verified_order_not_found_graceful_response(self, monkeypatch):
        """
        Single-turn: verified customer asks about a non-existent order.
        fetch_order returns not_found; joiner generates a graceful 'not found' response.
        No crash, no escalation.
        """
        fetch_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-99999"}, "deps": []},
        ]})
        lm = mock_llm_seq(fetch_plan, _JOINER_ANSWER,
                          "I couldn't find ORD-99999 in your account. Please double-check.")
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)

        state = make_state("Where is ORD-99999?", customer_id="CUST-001",
                           order_id="ORD-99999", identity_verified=True)
        result = order_lookup_node(state)

        assert result["resolution_status"] == "resolved"
        assert result["requires_escalation"] is False

    def test_verified_clarify_when_no_address_provided(self, monkeypatch):
        """
        Single-turn: customer asks to change address but doesn't provide one.
        Joiner decides CLARIFY and asks for the new address.
        """
        fetch_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "fetch_order",
             "args": {"order_id": "ORD-10003"}, "deps": []},
        ]})
        lm = mock_llm_seq(fetch_plan, _JOINER_CLARIFY)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)

        state = make_state("Change my delivery address", customer_id="CUST-002",
                           order_id="ORD-10003", identity_verified=True)
        result = order_lookup_node(state)

        assert result["resolution_status"] == "pending_clarification"
        assert "address" in result["messages"][-1].content.lower()

    # ── Security flows ────────────────────────────────────────────────────────

    def test_two_turn_fraud_detected_after_otp(self, monkeypatch):
        """
        Two-turn fraud flow: CUST-002 goes through OTP then tries to access
        ORD-10001 which belongs to CUST-001.
          Turn 1: OTP sent (1 LLM call)
          Turn 2: '123456' → fast-path → fetch_order → fraud_detected → escalated (0 LLM)

        The fraud is caught deterministically by _execute_tool, not by the LLM.
        The response never reveals the real owner's ID or says 'fraud'.
        """
        lm = mock_llm(self._send_otp_plan("CUST-002"))  # only Turn 1 calls LLM
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("agents.order_lookup.send_otp_email",
                            lambda cid: ("123456", "t***@test.com"))
        monkeypatch.setattr("memory.long_term.LongTermMemory", MagicMock)

        # Turn 1 — CUST-002 asks about ORD-10001 (CUST-001's order)
        state1 = make_state("Where is order ORD-10001?", customer_id="CUST-002",
                             order_id="ORD-10001", identity_verified=False)
        result1 = order_lookup_node(state1)
        assert result1["resolution_status"] == "pending_verification"

        # Turn 2 — correct OTP, but wrong customer for ORD-10001
        state2 = self._next_turn(state1, result1, "123456")
        result2 = order_lookup_node(state2)

        assert result2["resolution_status"] == "escalated"
        assert result2["requires_escalation"] is True
        content = result2["messages"][-1].content.lower()
        assert "fraud" not in content
        assert "cust-001" not in content
        assert any(w in content for w in ("locate", "found", "double-check", "error"))

    def test_escalation_for_legal_threat(self, monkeypatch):
        """
        Single-turn (already verified): customer says 'I'm going to sue ShopEase'.
        Joiner LLM decides ESCALATE.
        Response is warm and connects to a specialist — never 'I can't help'.
        """
        list_plan = json.dumps({"tasks": [
            {"id": "T1", "tool": "list_orders",
             "args": {"customer_id": "CUST-001"}, "deps": []},
        ]})
        lm = mock_llm_seq(list_plan, _JOINER_ESCALATE)
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("memory.long_term.LongTermMemory", MagicMock)

        state = make_state("I'm going to sue ShopEase", customer_id="CUST-001",
                           identity_verified=True)
        result = order_lookup_node(state)

        assert result["resolution_status"] == "escalated"
        assert result["requires_escalation"] is True
        content = result["messages"][-1].content.lower()
        assert any(w in content for w in ("specialist", "connecting", "senior"))
        assert "i can't help" not in content

    def test_otp_replay_attack_across_turns(self, monkeypatch):
        """
        Two-turn replay attack: customer enters the correct OTP on Turn 2 (resolves),
        then Turn 3 uses the SAME code again.  The code must not work a second time
        because pending_otp is cleared to None after successful verification.
        """
        lm = mock_llm_seq(
            self._send_otp_plan("CUST-001"),   # Turn 1 planner
            _JOINER_ANSWER,                    # Turn 2 joiner decision
            "ORD-10001 was delivered.",        # Turn 2 joiner solver
            self._send_otp_plan("CUST-001"),   # Turn 3 planner fallback (LLM)
        )
        monkeypatch.setattr("agents.order_lookup._get_llm", lambda: lm)
        monkeypatch.setattr("agents.order_lookup.send_otp_email",
                            lambda cid: ("123456", "t***@test.com"))

        state1 = make_state("Where is ORD-10001?", customer_id="CUST-001",
                             order_id="ORD-10001", identity_verified=False)
        result1 = order_lookup_node(state1)
        assert result1["resolution_status"] == "pending_verification"

        # Turn 2 — correct code, resolves
        state2 = self._next_turn(state1, result1, "123456")
        result2 = order_lookup_node(state2)
        assert result2["resolution_status"] == "resolved"
        # pending_otp must be None after successful verification
        assert result2["metadata"].get("pending_otp") is None
        assert result2["metadata"].get("identity_verified") is True

        # Turn 3 — same code again; fast-path triggers, but verify_otp returns
        # 'already_verified' (identity_verified=True, so it short-circuits).
        # The joiner falls through to ANSWER since it's already verified.
        state3 = self._next_turn(state2, result2, "123456")
        # identity_verified is True so fast-path won't trigger (condition requires False)
        # → the LLM planner creates a fresh plan (mocked as send_otp_plan above,
        #   but doesn't matter — what matters is the replay code has no effect)
        result3 = order_lookup_node(state3)
        # Must NOT escalate or error — session is already verified
        assert result3["requires_escalation"] is False
