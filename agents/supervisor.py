
"""
Node 4 — Supervisor / Orchestrator

Responsibilities:
  1. Classify customer intent from the conversation
  2. Extract key entities (order_id, refund_amount, product references)
  3. Route to the appropriate specialist agent
  4. Recall relevant long-term memory to inject context
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM = """\
You are the Supervisor of ShopEase's AI customer support system.
Your ONLY job is to analyse the customer's message and produce a routing decision —
you do NOT respond to the customer directly.

Available specialist agents:
- order_lookup   : Order status, tracking, delivery updates about a
                   SPECIFIC named order (ORD-XXXXX). Only use when
                   the customer is asking about a specific order they
                   placed. Do NOT use for general shipping questions.

- policy_returns : Return requests, refund requests, exchange requests,
                   warranty claims, questions about return/refund policy.
                   Use when customer explicitly says "return", "refund",
                   "exchange", or "money back".

- escalation     : ONLY use for explicit complaints, legal threats,
                   requests to speak to a manager, fraud reports, or
                   account security issues.
                   Examples that ARE escalation:
                     "I want to speak to a manager"
                     "I am going to sue ShopEase"
                     "someone hacked my account"
                     "this is unacceptable I demand compensation"
                   Examples that are NOT escalation:
                     "it keeps burning" → general (troubleshooting)
                     "it won't turn on" → general (troubleshooting)
                     "it makes a weird noise" → general (troubleshooting)
                     "it stopped working" → general (troubleshooting)
                     "I can't connect it" → general (troubleshooting)
                   CRITICAL: A customer describing a product problem
                   or malfunction is NOT an escalation — it is a
                   general troubleshooting question even if the wording
                   sounds urgent or concerning.

- general        : Everything else — product questions, troubleshooting,
                   product features, recommendations, store info,
                   promotions, account help, and ANY message that
                   describes a product issue or malfunction without
                   explicitly demanding manager involvement.
                   When in doubt, route to general.

Entity extraction rules:
- order_id: Look for patterns like ORD-XXXXX, #XXXXX, "order 12345", "order number 10001"
- refund_amount: Any dollar amount mentioned alongside words like "refund", "return", "money back"

Respond with ONLY valid JSON — no extra text, no markdown fences:
{{
  "intent": "<order_lookup|policy_returns|escalation|general>",
  "order_id": "<extracted order ID or null>",
  "refund_amount": <float or null>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence — cite the specific words that determined the intent, e.g. 'customer said burning which is troubleshooting not escalation'>"
}}
"""


_llm_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        from langchain_groq import ChatGroq
        _llm_instance = ChatGroq(
            model=settings.model_name,
            temperature=0.0,
            api_key=settings.groq_api_key,
        )
    return _llm_instance


def _extract_conversation_text(messages: list) -> str:
    """Flatten the last N messages into a plain text block."""
    lines = []
    for m in messages[-6:]:  # last 6 turns for context
        role = getattr(m, "type", "unknown")
        if role == "human":
            lines.append(f"Customer: {m.content}")
        elif role == "ai":
            lines.append(f"Agent: {m.content}")
    return "\n".join(lines)


def _fallback_intent(text: str) -> str:
    """Cheap regex fallback if the LLM fails."""
    text_lower = text.lower()
    if re.search(r"\b(order|track|ship|deliver|package|status|where is)\b", text_lower):
        return "order_lookup"
    if re.search(r"\b(return|refund|exchange|money back|cancel|broken|defect)\b", text_lower):
        return "policy_returns"
    if re.search(r"\b(manager|escalate|complain|unacceptable|lawsuit|horrible)\b", text_lower):
        return "escalation"
    return "general"


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
        "question": invoke_kwargs.get("question", invoke_kwargs.get("conversation", "")),
        "context_summary": invoke_kwargs.get("policy_context") or invoke_kwargs.get("order_info", ""),
        "initial_answer": initial_answer,
    })


def supervisor_node(state: dict) -> dict:
    """
    LangGraph node — Supervisor/Orchestrator.

    Reads: messages, customer_id, session_id
    Writes: intent, order_id, refund_amount, next_agent, start_time
    """
    start_time = time.time()
    messages = state.get("messages", [])
    customer_id = state.get("customer_id", "unknown")
    session_id = state.get("session_id", "unknown")

    conversation_text = _extract_conversation_text(messages)

    # ── LLM classification ────────────────────────────────────────────────────
    intent = "general"
    order_id = state.get("order_id")  # preserve if already extracted
    refund_amount: Optional[float] = None
    confidence = 0.5

    try:
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYSTEM),
            ("human", "Customer conversation:\n{conversation}"),
        ])
        chain = prompt | llm | StrOutputParser()
        invoke_kwargs = {"conversation": conversation_text}
        raw = chain.invoke(invoke_kwargs)
        if _needs_reflection(raw):
            logger.debug("Supervisor: self-reflection triggered")
            try:
                raw = _reflect(invoke_kwargs, initial_answer=raw, llm=llm)
            except Exception as ref_e:
                logger.warning("Supervisor: reflection failed: %s", ref_e)

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        parsed = json.loads(raw)

        intent = parsed.get("intent", "general")
        extracted_order_id = parsed.get("order_id")
        if extracted_order_id and extracted_order_id != "null":
            order_id = extracted_order_id
        if parsed.get("refund_amount"):
            try:
                refund_amount = float(parsed["refund_amount"])
            except (ValueError, TypeError):
                pass
        confidence = float(parsed.get("confidence", 0.5))

        logger.info(
            "Supervisor | intent=%s order_id=%s confidence=%.2f | customer=%s session=%s",
            intent, order_id, confidence, customer_id, session_id,
        )

    except Exception as e:
        logger.warning("Supervisor LLM call failed (%s) — using regex fallback.", e)
        last_human = next(
            (m.content for m in reversed(messages) if getattr(m, "type", "") == "human"),
            "",
        )
        intent = _fallback_intent(last_human)

    # ── Long-term memory recall ───────────────────────────────────────────────
    past_context = ""
    try:
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory()
        if ltm.get_customer_history_count(customer_id) > 0:
            last_human_text = next(
                (m.content for m in reversed(messages) if getattr(m, "type", "") == "human"),
                "",
            )
            recall_query = f"{intent} {order_id or ''} {last_human_text[:100]}".strip()
            memories = ltm.recall(customer_id=customer_id, query=recall_query)
            past_context = ltm.format_for_prompt(memories)
    except Exception as e:
        logger.debug("Long-term memory recall skipped: %s", e)

    return {
        "intent": intent,
        "order_id": order_id,
        "refund_amount": refund_amount,
        "next_agent": intent,
        "start_time": start_time,
        "turn_count": state.get("turn_count", 0) + 1,
        "metadata": {
            **state.get("metadata", {}),
            "supervisor_confidence": confidence,
            "past_context": past_context,
        },
    }
