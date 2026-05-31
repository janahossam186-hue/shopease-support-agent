"""
Node 3 — Escalation Agent

Handles:
  • Complaints that couldn't be resolved by other agents
  • Requests to speak with a manager
  • Policy exceptions (returns > 30 days, refunds > $500)
  • Urgent/critical issues

Workflow:
  1. Generate a unique ticket ID
  2. Log the escalation details
  3. Generate an empathetic response with ticket number + ETA
  4. Save escalation summary to long-term memory
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

_long_term_memory = LongTermMemory()

ESCALATION_AGENT_SYSTEM = """\
You are a Senior Support Specialist at ShopEase with authority to handle escalations.
You have been brought in because the standard resolution path wasn't sufficient.

Your tone must be:
- Highly empathetic and apologetic
- Professional and reassuring
- Specific about next steps and timelines

Always:
- Acknowledge the customer's frustration sincerely
- Provide the ticket ID prominently
- Give a realistic resolution ETA (see context)
- Explain what will happen next (supervisor will call / email within N hours)
- Offer a gesture of goodwill if appropriate (e.g., "We'll make sure this is resolved.")
- Do NOT over-promise specific outcomes (like a guaranteed refund) without authority

Keep the response warm and under 4 paragraphs.
"""

ESCALATION_AGENT_HUMAN = """\
Customer ID: {customer_id}
Escalation Ticket ID: {ticket_id}
Escalation Reason: {escalation_reason}
Estimated Resolution: {eta}
RMA Number: {rma_number}

Order Information (if applicable):
{order_info}

Policy Violations / Notes:
{policy_notes}

Past Customer Context:
{past_context}

Conversation History:
{history}

Customer's Message: {question}
"""


_llm_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        from langchain_groq import ChatGroq
        _llm_instance = ChatGroq(
            model=settings.model_name,
            temperature=0.2,
            api_key=settings.groq_api_key,
        )
    return _llm_instance


def _generate_ticket_id() -> str:
    """Generate a ticket ID like TKT-A3F9XB2C."""
    suffix = uuid.uuid4().hex[:8].upper()
    return f"TKT-{suffix}"


def _get_history_text(messages: list) -> str:
    lines = []
    for m in messages:
        role = getattr(m, "type", "unknown")
        if role == "human":
            lines.append(f"Customer: {m.content}")
        elif role == "ai":
            lines.append(f"Agent: {m.content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _needs_reflection(answer: str) -> bool:
    """Return True if the answer is too short or expresses uncertainty."""
    if len(answer.strip()) < 20:
        return True
    uncertainty_phrases = [
        "i don't know", "i'm not sure", "i cannot",
        "unclear", "no information", "not mentioned",
    ]
    return any(phrase in answer.lower() for phrase in uncertainty_phrases)


def _reflect(invoke_kwargs: dict, initial_answer: str, llm) -> str:
    """Second-pass prompt that shows the model its own answer for correction."""
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are reviewing your own previous answer for accuracy and completeness."),
        ("human", (
            "Original question: {question}\n\n"
            "Context you had: {context_summary}\n\n"
            "Your initial answer: {initial_answer}\n\n"
            "If the answer is incomplete or uncertain, improve it. "
            "If it is correct and complete, repeat it unchanged."
        )),
    ])
    chain = reflection_prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": invoke_kwargs["question"],
        "context_summary": invoke_kwargs.get("policy_context") or invoke_kwargs.get("order_info", ""),
        "initial_answer": initial_answer,
    })


def _determine_escalation_reason(state: dict) -> str:
    """Build a human-readable escalation reason from state."""
    reasons = []
    intent = state.get("intent", "")
    violations = state.get("policy_violations", [])
    requires_escalation = state.get("requires_escalation", False)

    if intent == "escalation":
        reasons.append("Customer requested to speak with a manager")
    if violations:
        for v in violations:
            reasons.append(v)
    elif requires_escalation:
        reasons.append("Request falls outside standard agent authority")

    return "; ".join(reasons) if reasons else "Escalated from customer request"


def _determine_eta(state: dict) -> str:
    """Return a realistic resolution timeline."""
    refund_amount = state.get("refund_amount")
    if refund_amount and refund_amount > 1000:
        return "2–3 business days (manager review required for large refunds)"
    if refund_amount and refund_amount > 500:
        return "1 business day (supervisor approval needed)"
    violations = state.get("policy_violations", [])
    if any("return window" in v.lower() or "POL-002" in v for v in violations):
        return "2–3 business days (return window exception requires manager approval)"
    return "4 business hours"


def escalation_node(state: dict) -> dict:
    """
    LangGraph node — Escalation Agent.

    Reads: messages, customer_id, order_id, requires_escalation, metadata
    Writes: messages, agent_used, resolution_status, escalation_ticket_id
    """
    messages = state.get("messages", [])
    customer_id = state.get("customer_id", "unknown")
    order_id = state.get("order_id")
    metadata = state.get("metadata", {})
    past_context = metadata.get("past_context", "No prior interactions found.")

    last_human = next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "human"),
        "",
    )
    history_text = _get_history_text(messages)

    # Ticket generation
    ticket_id = _generate_ticket_id()
    escalation_reason = _determine_escalation_reason(state)
    eta = _determine_eta(state)

    # Order info
    order_info = "No specific order referenced."
    if order_id:
        try:
            from tools.order_tools import get_order_tool
            order_info = get_order_tool.invoke({"order_id": order_id})
        except Exception:
            pass

    # Policy notes
    policy_violations = state.get("policy_violations", [])
    policy_notes = "\n".join(f"• {v}" for v in policy_violations) if policy_violations else "None"

    # RMA number for return-related escalations
    is_return_escalation = any(
        "return window" in v.lower() or "POL-002" in v for v in policy_violations
    )
    rma_number = f"RMA-{uuid.uuid4().hex[:8].upper()}" if is_return_escalation else None

    # ── Generate response ─────────────────────────────────────────────────────
    try:
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", ESCALATION_AGENT_SYSTEM),
            ("human", ESCALATION_AGENT_HUMAN),
        ])
        chain = prompt | llm | StrOutputParser()
        invoke_kwargs = {
            "customer_id": customer_id,
            "ticket_id": ticket_id,
            "escalation_reason": escalation_reason,
            "eta": eta,
            "rma_number": rma_number or "N/A",
            "order_info": order_info,
            "policy_notes": policy_notes,
            "past_context": past_context,
            "history": history_text,
            "question": last_human,
        }
        response = chain.invoke(invoke_kwargs)
        if _needs_reflection(response):
            logger.debug("Escalation: self-reflection triggered")
            try:
                response = _reflect(invoke_kwargs, initial_answer=response, llm=llm)
            except Exception as ref_e:
                logger.warning("Escalation: reflection failed: %s", ref_e)
    except Exception as e:
        logger.error("Escalation agent LLM call failed: %s", e)
        response = (
            f"I'm escalating your case to our senior support team (Ticket: {ticket_id}). "
            f"A specialist will contact you within {eta}. "
            "We sincerely apologise for the inconvenience."
        )

    # ── Save escalation to long-term memory ───────────────────────────────────
    try:
        session_id = state.get("session_id", "unknown")
        summary = (
            f"Escalation ticket {ticket_id} created. "
            f"Reason: {escalation_reason}. "
            f"Order: {order_id or 'N/A'}. "
            f"Customer complaint: {last_human}. "
            f"Customer requested manager review. ETA: {eta}."
        )
        _long_term_memory.save_interaction(
            customer_id=customer_id,
            session_id=session_id,
            summary=summary,
            metadata={
                "intent": "escalation",
                "ticket_id": ticket_id,
                "resolution": "escalated",
            },
        )
    except Exception as e:
        logger.warning("Failed to save escalation to long-term memory: %s", e)

    logger.info(
        "Escalation Agent | ticket=%s customer=%s reason=%s",
        ticket_id, customer_id, escalation_reason[:60],
    )

    result = {
        "messages": [AIMessage(content=response)],
        "agent_used": "escalation",
        "resolution_status": "escalated",
        "escalation_ticket_id": ticket_id,
    }
    if rma_number:
        result["rma_number"] = rma_number
    return result
