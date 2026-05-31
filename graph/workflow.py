"""
LangGraph Workflow — E-Commerce Customer Support Agent

Graph architecture:
                         ┌──────────────────────┐
                         │   input_guardrail    │ ← checks prompt injection
                         └──────────┬───────────┘
                             pass   │   block → END (with rejection message)
                                    ▼
                         ┌──────────────────────┐
                         │      supervisor      │ ← classifies intent, routes
                         └──────┬───────┬───────┘
                order_lookup    │       │ policy_returns    │ escalation / general
                                ▼       ▼                   ▼
               ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
               │  order_lookup    │  │ policy_returns   │  │   escalation     │
               └────────┬─────────┘  └───────┬──────────┘  └────────┬─────────┘
                        │                    │ needs escalation       │
                        │                    └────────────────────────┤
                        │                                             │
                        └──────────────────┬──────────────────────────┘
                                           ▼
                              ┌──────────────────────┐
                              │   output_guardrail   │ ← PII redaction, toxicity
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │   finalize_metrics   │ ← latency logging
                              └──────────┬───────────┘
                                         ▼
                                        END
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


# ── State definition ──────────────────────────────────────────────────────────

class CustomerSupportState(TypedDict):
    # ── Conversation (LangGraph manages append-only via add_messages) ──────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Session context ────────────────────────────────────────────────────
    customer_id: str
    session_id: str
    turn_count: int

    # ── Request analysis ───────────────────────────────────────────────────
    order_id: Optional[str]
    refund_amount: Optional[float]
    intent: str
    next_agent: str

    # ── RAG context ────────────────────────────────────────────────────────
    retrieved_docs: list[dict]
    retrieval_scores: list[float]

    # ── Guardrail state ────────────────────────────────────────────────────
    guardrail_passed: bool
    guardrail_reason: str
    toxicity_score: float

    # ── Policy state ───────────────────────────────────────────────────────
    policy_compliant: bool
    requires_escalation: bool

    # ── Resolution tracking ────────────────────────────────────────────────
    resolution_status: str           # pending | resolved | escalated | blocked
    escalation_ticket_id: Optional[str]
    agent_used: str

    # ── Evaluation ────────────────────────────────────────────────────────
    start_time: float
    latency_ms: float
    metadata: dict


# ── Default state factory ──────────────────────────────────────────────────────

def make_initial_state(
    customer_id: str,
    session_id: str,
    user_message: str,
) -> dict:
    """
    Build the initial state dict for a new conversation turn.
    Inject a HumanMessage so the graph has something to work with.
    """
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


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_guardrail(state: CustomerSupportState) -> Literal["supervisor", "__end__"]:
    """Block injections; pass everything else to the supervisor."""
    if state.get("guardrail_passed", True):
        return "supervisor"
    return END


def route_after_supervisor(state: CustomerSupportState) -> str:
    """Map the classified intent to an agent node name."""
    intent_map = {
        "order_lookup": "order_lookup",
        "policy_returns": "policy_returns",
        "escalation": "escalation",
        "general": "general",      # routed to the general agent (Layla)
        "unknown": "general",      # unknown intent → general agent as safe default
    }
    return intent_map.get(state.get("intent", "unknown"), "general")


def route_after_policy(state: CustomerSupportState) -> str:
    """Escalate if policy guardrail triggered, otherwise go to output check."""
    if state.get("requires_escalation", False):
        return "escalation"
    return "output_guardrail"


# ── Metrics finaliser node ─────────────────────────────────────────────────────

def finalize_metrics_node(state: CustomerSupportState) -> dict:
    """Calculate latency and persist evaluation metrics."""
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


# ── Graph construction ────────────────────────────────────────────────────────

def create_graph():
    """Build and compile the LangGraph StateGraph with full checkpointing."""
    builder = StateGraph(CustomerSupportState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("input_guardrail", input_guardrail_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("order_lookup", order_lookup_node)
    builder.add_node("policy_returns", policy_returns_node)
    builder.add_node("escalation", escalation_node)
    builder.add_node("general", general_agent_node)
    builder.add_node("output_guardrail", output_guardrail_node)
    builder.add_node("finalize_metrics", finalize_metrics_node)

    # ── Edges ─────────────────────────────────────────────────────────────
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

    # order_lookup → output_guardrail (always)
    builder.add_edge("order_lookup", "output_guardrail")

    # general → output_guardrail (always)
    builder.add_edge("general", "output_guardrail")

    # policy_returns → escalation OR output_guardrail
    builder.add_conditional_edges(
        "policy_returns",
        route_after_policy,
        {
            "escalation": "escalation",
            "output_guardrail": "output_guardrail",
        },
    )

    # escalation → output_guardrail (always)
    builder.add_edge("escalation", "output_guardrail")

    # output_guardrail → finalize → END
    builder.add_edge("output_guardrail", "finalize_metrics")
    builder.add_edge("finalize_metrics", END)

    # ── Compile with short-term memory checkpointer ───────────────────────
    checkpointer = get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("LangGraph workflow compiled successfully.")
    return graph


# ── Singleton graph instance ──────────────────────────────────────────────────

_graph = None


def get_graph():
    """Return the singleton compiled graph, creating it if needed."""
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph
