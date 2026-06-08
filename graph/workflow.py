"""
LangGraph Workflow — E-Commerce Customer Support Agent
"""
from __future__ import annotations

import logging
import time
from typing import Annotated, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agents.escalation import escalation_node
from agents.general_agent import general_agent_node
from agents.order_lookup import order_lookup_node
from agents.policy_returns import policy_returns_node
from agents.supervisor import supervisor_node
from guardrails.input_guardrail import input_guardrail_node
from guardrails.toxicity_guardrail import output_guardrail_node
from memory.short_term import get_checkpointer
from evaluation.metrics import log_interaction

logger = logging.getLogger(__name__)


class CustomerSupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: str
    session_id: str
    turn_count: int
    order_id: Optional[str]
    refund_amount: Optional[float]
    intent: str
    next_agent: str
    retrieved_docs: list[dict]
    retrieval_scores: list[float]
    guardrail_passed: bool
    guardrail_reason: str
    toxicity_score: float
    policy_compliant: bool
    requires_escalation: bool
    resolution_status: str
    escalation_ticket_id: Optional[str]
    agent_used: str
    start_time: float
    latency_ms: float
    metadata: Annotated[dict, lambda old, new: {**old, **new}]


def make_initial_state(
    customer_id: str,
    session_id: str,
    user_message: str,
) -> dict:
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=user_message)],
        "customer_id": customer_id,
        "session_id": session_id,
        "turn_count": 0,
        "order_id": None,
        "refund_amount": None,
        "intent": "unknown",
        "next_agent": "unknown",
        "retrieved_docs": [],
        "retrieval_scores": [],
        "guardrail_passed": True,
        "guardrail_reason": "",
        "toxicity_score": 0.0,
        "policy_compliant": True,
        "requires_escalation": False,
        "resolution_status": "pending",
        "escalation_ticket_id": None,
        "agent_used": "none",
        "start_time": time.time(),
        "latency_ms": 0.0,
        "metadata": {},
    }


def route_after_guardrail(state: CustomerSupportState) -> Literal["supervisor", "__end__"]:
    if state.get("guardrail_passed", True):
        return "supervisor"
    return END


def route_after_supervisor(state: CustomerSupportState) -> str:
    intent_map = {
        "order_lookup": "order_lookup",
        "policy_returns": "policy_returns",
        "escalation": "escalation",
        "general": "general",
        "unknown": "general",
    }
    return intent_map.get(state.get("intent", "unknown"), "general")


def route_after_agent(state: CustomerSupportState) -> str:
    if state.get("requires_escalation", False):
        return "escalation"
    if state.get("resolution_status") == "needs_rerouting":
        return "supervisor"
    return "output_guardrail"


def finalize_metrics_node(state: CustomerSupportState) -> dict:
    start = state.get("start_time", time.time())
    latency_ms = (time.time() - start) * 1000
    try:
        log_interaction(
            session_id=state.get("session_id", "unknown"),
            customer_id=state.get("customer_id", "unknown"),
            intent=state.get("intent", "unknown"),
            agent_used=state.get("agent_used", "unknown"),
            resolution_status=state.get("resolution_status", "pending"),
            latency_ms=latency_ms,
            guardrail_passed=state.get("guardrail_passed", True),
            retrieved_doc_count=len(state.get("retrieved_docs", [])),
            policy_compliant=state.get("policy_compliant", True),
            avg_retrieval_score=(
                sum(state.get("retrieval_scores", [0])) / len(state.get("retrieval_scores", [1]))
                if state.get("retrieval_scores") else 0.0
            ),
            toxicity_score=state.get("toxicity_score", 0.0),
        )
    except Exception as e:
        logger.warning("Failed to log interaction metrics: %s", e)
    return {"latency_ms": latency_ms}


def create_graph():
    builder = StateGraph(CustomerSupportState)

    builder.add_node("input_guardrail", input_guardrail_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("order_lookup", order_lookup_node)
    builder.add_node("policy_returns", policy_returns_node)
    builder.add_node("escalation", escalation_node)
    builder.add_node("general", general_agent_node)
    builder.add_node("output_guardrail", output_guardrail_node)
    builder.add_node("finalize_metrics", finalize_metrics_node)

    builder.add_edge(START, "input_guardrail")

    builder.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"supervisor": "supervisor", END: END},
    )

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "order_lookup": "order_lookup",
            "policy_returns": "policy_returns",
            "escalation": "escalation",
            "general": "general",
        },
    )

    for _agent in ("order_lookup", "policy_returns", "general"):
        builder.add_conditional_edges(
            _agent,
            route_after_agent,
            {
                "supervisor": "supervisor",
                "escalation": "escalation",
                "output_guardrail": "output_guardrail",
            },
        )

    builder.add_edge("escalation", "output_guardrail")
    builder.add_edge("output_guardrail", "finalize_metrics")
    builder.add_edge("finalize_metrics", END)

    checkpointer = get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("LangGraph workflow compiled successfully.")
    return graph


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph