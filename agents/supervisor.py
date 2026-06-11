
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


def _decompose(conversation_text: str, llm) -> list[str] | None:
    """
    Detect if the customer message contains multiple intents.
    Returns a list of intents in the order they should be handled.
    Returns None if only one intent is detected.
    """
    prompt = f"""You are analyzing a customer message for ShopEase \
Egypt's support system.

Determine if this message contains MORE THAN ONE distinct request \
that requires different specialist agents.

Available agents and what they handle:

- order_lookup   : ANY question about a specific order — tracking, \
                   delivery status, estimated arrival, shipping \
                   carrier, order modifications (cancel, update \
                   address, change quantity), return initiation \
                   that requires fetching order details first.
                   Requires an order ID or customer order history.

- policy_returns : Return eligibility, refund requests, exchange \
                   requests, return window rules, non-returnable \
                   items, restocking fees, warranty claims, \
                   questions about what the return policy says.
                   Does NOT need an order ID — handles policy \
                   questions even without a specific order.

- general        : Product features, specs, pricing, stock \
                   availability, product manuals, troubleshooting, \
                   store locations, payment methods, promotions, \
                   shipping policy questions (general, not about \
                   a specific order), skincare recommendations, \
                   account help, any question not covered above.

- escalation     : Explicit complaints, legal threats ("I will sue"),
                   requests to speak to a manager, suspected fraud, \
                   account compromise, situations where the customer \
                   is extremely angry or the issue is unresolvable \
                   by automated agents.
                   NEVER use escalation for product questions or \
                   order tracking — only for genuine escalation needs.

Rules:
- Only decompose if the message CLEARLY contains 2+ distinct \
  requests requiring DIFFERENT agents
- "Track my order AND tell me if I can return it" → 2 intents:
  ["order_lookup", "policy_returns"]
- "Where is my order?" → 1 intent, do NOT decompose:
  ["order_lookup"]
- "What is the return policy for electronics?" → 1 intent:
  ["policy_returns"]
- "Show me my order status and what headphones you have" → 2 intents:
  ["order_lookup", "general"]
- Maximum 3 intents per message
- Order matters: put order_lookup before policy_returns if both \
  present, since order details may be needed for returns

IMPORTANT: Only include order_lookup if the customer's \
CURRENT message explicitly mentions an order, tracking, \
delivery, or shipping for a specific order. Do NOT include \
order_lookup just because an order ID exists in conversation \
history from a previous turn.

Customer conversation:
{conversation_text}

Respond with ONLY a JSON array of intent strings, nothing else.
No explanation, no markdown, no extra text.
Examples of valid responses:
["order_lookup"]
["order_lookup", "policy_returns"]
["general", "policy_returns"]
"""
    try:
        raw = llm.invoke(prompt).content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        intents = json.loads(raw)
        if isinstance(intents, list) and len(intents) > 0:
            valid = {"order_lookup", "policy_returns", "general", "escalation"}
            intents = [i for i in intents if i in valid]
            if len(intents) > 1:
                return intents
    except Exception as e:
        logger.warning("Decomposition failed: %s — skipping", e)
    return None


def supervisor_node(state: dict) -> dict:
    """
    LangGraph node — Supervisor/Orchestrator.

    Reads: messages, customer_id, session_id
    Writes: intent, order_id, refund_amount, next_agent, start_time
    """
    # Reset decomposition state for new queries
    last_resolution = state.get("resolution_status", "pending")
    if (not state.get("pending_intents") and
        last_resolution in ("resolved", "escalated", "blocked", "pending")):
        # This is a fresh query — clear any leftover
        # decomposition state from previous turns
        partial_responses = []
        accumulated_docs = []
        # Also reset is_decomposed
        is_decomposed_flag = False
    else:
        partial_responses = list(state.get("partial_responses", []))
        accumulated_docs = list(state.get("accumulated_docs", []))
        is_decomposed_flag = state.get("is_decomposed", False)

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

    # ── Query decomposition ───────────────────────────────────────────────────
    pending_intents = list(state.get("pending_intents", []))

    last_human = next(
        (m.content for m in reversed(messages)
         if getattr(m, "type", "") == "human"), ""
    )
    is_otp_response = bool(
        re.fullmatch(r"\s*\d{6}\s*", last_human.strip())
    )

    if is_otp_response:
        intent = "order_lookup"
        pending_intents = []
        logger.info(
            "Supervisor: OTP response detected — routing to "
            "order_lookup, skipping decomposition"
        )
    elif not pending_intents:
        try:
            decomposed = _decompose(conversation_text, _get_llm())
            if decomposed and len(decomposed) > 1:
                logger.info(
                    "Supervisor: decomposed into %d intents: %s",
                    len(decomposed), decomposed,
                )
                intent = decomposed[0]
                pending_intents = decomposed[1:]
        except Exception as e:
            logger.warning("Decomposition step failed: %s", e)
    else:
        intent = pending_intents[0]
        pending_intents = pending_intents[1:]
        logger.info(
            "Supervisor: continuing decomposed request → %s "
            "(%d remaining)", intent, len(pending_intents),
        )

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

    # Accumulate retrieved docs from previous agents
    current_docs = state.get("retrieved_docs", [])
    if current_docs and (pending_intents or state.get("is_decomposed", False)):
        accumulated_docs.extend(current_docs)

    # Save last AI response to partial_responses if this is part of a decomposed request
    if pending_intents or state.get("is_decomposed", False):
        last_ai = next(
            (m.content for m in reversed(messages)
             if getattr(m, "type", "") == "ai"),
            "",
        )
        if last_ai and last_ai not in partial_responses:
            partial_responses.append(last_ai)

    return {
        "intent": intent,
        "order_id": order_id,
        "refund_amount": refund_amount,
        "next_agent": intent,
        "start_time": start_time,
        "turn_count": state.get("turn_count", 0) + 1,
        "pending_intents": pending_intents,
        "partial_responses": partial_responses,
        "is_decomposed": len(pending_intents) > 0 or is_decomposed_flag,
        "accumulated_docs": accumulated_docs,
        "metadata": {
            **state.get("metadata", {}),
            "supervisor_confidence": confidence,
            "past_context": past_context,
        },
    }
