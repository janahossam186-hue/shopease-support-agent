"""Quick smoke-test for the two-turn OTP verification flow."""
import sys
sys.path.insert(0, ".")

from unittest.mock import patch
from langchain_core.messages import HumanMessage
from agents.order_lookup import order_lookup_node

FAKE_OTP = "999888"


def make_state(msg, customer_id="CUST-001", order_id=None, identity_verified=False):
    return {
        "messages": [HumanMessage(content=msg)],
        "customer_id": customer_id,
        "session_id": "smoke-test",
        "order_id": order_id,
        "agent_used": "",
        "resolution_status": "",
        "requires_escalation": False,
        "retrieved_docs": [],
        "retrieval_scores": [],
        "metadata": {"identity_verified": identity_verified},
    }


def next_turn(prev_state, prev_result, new_msg):
    merged_meta = {**prev_state.get("metadata", {}), **prev_result.get("metadata", {})}
    return {
        **prev_state,
        **{k: v for k, v in prev_result.items() if k not in ("messages", "metadata")},
        "messages": (
            prev_state.get("messages", [])
            + prev_result.get("messages", [])
            + [HumanMessage(content=new_msg)]
        ),
        "metadata": merged_meta,
    }


with patch("agents.order_lookup.send_otp_email", return_value=(FAKE_OTP, "t***@test.com")):

    # ── Turn 1 ─────────────────────────────────────────────────────────────────
    print("=== TURN 1: customer asks about orders ===")
    state1 = make_state("Show me my orders", customer_id="CUST-001")
    result1 = order_lookup_node(state1)
    print(f"  resolution_status : {result1['resolution_status']}")
    print(f"  pending_otp stored: {result1['metadata'].get('pending_otp')}")
    print(f"  agent reply       : {result1['messages'][-1].content[:100]}")

    assert result1["resolution_status"] == "pending_verification", "FAIL: expected pending_verification"
    assert result1["metadata"].get("pending_otp") == FAKE_OTP, "FAIL: OTP not stored in metadata"
    print("  [PASS] OTP sent and stored correctly")

    # ── Turn 2: customer enters OTP ────────────────────────────────────────────
    print()
    print("=== TURN 2: customer enters OTP ===")
    state2 = next_turn(state1, result1, FAKE_OTP)
    print(f"  pending_otp in state  : {state2['metadata'].get('pending_otp')}")
    print(f"  identity_verified     : {state2['metadata'].get('identity_verified')}")

    result2 = order_lookup_node(state2)
    print(f"  identity_verified after: {result2['metadata'].get('identity_verified')}")
    print(f"  resolution_status      : {result2['resolution_status']}")
    print(f"  agent reply (first 200): {result2['messages'][-1].content[:200]}")

    assert result2["metadata"].get("identity_verified") is True, "FAIL: identity not verified"
    assert result2["resolution_status"] == "resolved", "FAIL: expected resolved"
    print("  [PASS] OTP verified — identity confirmed — orders returned")
