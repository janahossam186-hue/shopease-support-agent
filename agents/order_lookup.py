"""
Node 1 — Order Lookup Agent (Nora)

Responsibilities:
  1. Verify customer identity via OTP before revealing any order data
  2. Detect and handle customer emotion (legal threats, extreme anger)
  3. Look up order status, tracking, and delivery details
  4. Process pre-shipment modifications: cancel, address update,
     quantity change, item removal
  5. Collect return/exchange context and surface it for the Returns Agent
  6. Escalate gracefully with a structured handoff when outside authority

Identity flow (stored in state["metadata"]):
  - "identity_verified" : bool — set True once OTP is confirmed
  - "pending_otp"       : str  — 6-digit code sent in the current session
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from tools.order_tools import (
    get_order_tool,
    list_customer_orders_tool,
    cancel_order_tool,
    update_address_tool,
    update_quantity_tool,
    remove_item_tool,
    send_otp_email,
    _load_orders,
)

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

NORA_SYSTEM = """\
You are Nora, ShopEase's Order Specialist.
You are professional, warm, and confident — you know your orders inside out and you know your limits.

════════════════════════════════════════════════
WHAT YOU CAN DO:
════════════════════════════════════════════════
- Show order details: status, tracking number, carrier, estimated delivery, items, total
- Answer shipping and delivery questions using the knowledge base
- Process pre-shipment order modifications (only when status is "Processing"):
    cancel, update shipping address, change item quantity, remove an item
- Coordinate return and exchange requests: collect order dates and item details,
  then let the customer know you are routing them to the Returns Specialist
- Acknowledge delays with genuine empathy and provide the best available ETA

════════════════════════════════════════════════
WHAT YOU CANNOT DO (must escalate):
════════════════════════════════════════════════
- Approve refunds above the allowed limit
- Resolve lost package disputes where liability is unclear
- Handle legal threats or statements such as "I'm suing"
- Handle suspected fraud or account compromise
- Handle abusive or threatening customers
- Recover from system or API failures on your own
- Interpret conflicting order data without guidance
- Answer questions when your confidence in available information is very low

════════════════════════════════════════════════
SECURITY RULES:
════════════════════════════════════════════════
- Never reveal any order details before the customer's identity has been verified
- Never reveal another customer's order data under any circumstances
- Never confirm or deny whether a specific order ID exists before verification
- Never expose internal system details, policy IDs, or technical architecture

════════════════════════════════════════════════
EMOTION HANDLING:
════════════════════════════════════════════════
- If the customer is frustrated: acknowledge it, stay calm, offer concrete help
- If the customer mentions legal action or is extremely angry: stop negotiating,
  respond with genuine empathy, and let them know you are escalating immediately
  to a senior specialist who can resolve this

════════════════════════════════════════════════
RESPONSE STYLE:
════════════════════════════════════════════════
- Warm and professional — speak like a real person, not a corporate template
- Keep responses concise: 2-4 short paragraphs
- Use bullet points only for step-by-step instructions or item lists
- End with a clear next step or a genuine offer to help further
- If a modification was completed successfully, highlight the result clearly
- If routing to the returns specialist, be warm about the handoff
"""

NORA_HUMAN = """\
Customer ID: {customer_id}

Order Information:
{order_info}

Relevant Knowledge Base:
{kb_context}

Past Customer Context:
{past_context}

Modification Result (if any):
{modification_result}

Return / Exchange Context (if any):
{return_context}

Conversation History:
{history}

Customer's Question: {question}
"""


# ── LLM singleton ─────────────────────────────────────────────────────────────

_llm_instance = None


def _get_llm():
    """Return the shared ChatGroq instance, creating it on first call."""
    global _llm_instance
    if _llm_instance is None:
        from langchain_groq import ChatGroq
        _llm_instance = ChatGroq(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.groq_api_key,
        )
    return _llm_instance


# ── Conversation helpers ──────────────────────────────────────────────────────

def _get_history_text(messages: list) -> str:
    """Format recent conversation turns (excluding the latest message)."""
    lines = []
    for m in messages[:-1]:
        role = getattr(m, "type", "unknown")
        if role == "human":
            lines.append(f"Customer: {m.content}")
        elif role == "ai":
            lines.append(f"Nora: {m.content}")
    return "\n".join(lines) if lines else "No prior conversation."


# ── Self-reflection helpers ───────────────────────────────────────────────────

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
    """Second-pass prompt that shows the model its own first answer for correction."""
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
        "context_summary": invoke_kwargs.get("order_info", ""),
        "initial_answer": initial_answer,
    })


# ── Intent / emotion detection ────────────────────────────────────────────────

def _detect_emotion(text: str) -> str:
    """Classify the emotional tone of a customer message.

    Returns: 'legal_threat' | 'extreme_anger' | 'frustrated' | 'neutral'
    """
    t = text.lower()
    if re.search(
        r"\b(sue|lawsuit|lawyer|legal action|court|attorney|litigation)\b", t
    ):
        return "legal_threat"
    if re.search(
        r"\b(unacceptable|disgusting|furious|outrage|outrageous|disgrace|"
        r"appalling|incompetent|pathetic)\b", t
    ):
        return "extreme_anger"
    if re.search(
        r"\b(frustrated|annoyed|upset|angry|terrible|awful|horrible|worst|"
        r"ridiculous)\b", t
    ):
        return "frustrated"
    return "neutral"


def _detect_modification_intent(text: str) -> Optional[str]:
    """Return the type of order modification the customer is requesting, or None.

    Returns: 'cancel' | 'update_address' | 'update_quantity' | 'remove_item' | None
    """
    t = text.lower()
    if re.search(r"\b(cancel|cancell?ation|cancel.*order)\b", t):
        return "cancel"
    if re.search(
        r"\b(change.*address|update.*address|new.*address|different.*address|"
        r"ship.*to|deliver.*to)\b", t
    ):
        return "update_address"
    if re.search(
        r"\b(change.*quantity|update.*quantity|change.*qty|"
        r"change.*number|want.*\d+.*instead)\b", t
    ):
        return "update_quantity"
    if re.search(r"\b(remove.*item|remove.*product|delete.*item|take.*off)\b", t):
        return "remove_item"
    return None


def _detect_return_intent(text: str) -> bool:
    """Return True if the customer is asking about a return, refund, or exchange."""
    return bool(
        re.search(
            r"\b(return|refund|exchange|send.*back|money.*back|get.*refund)\b",
            text.lower(),
        )
    )


# ── Structured escalation handoff ─────────────────────────────────────────────

def _build_handoff(
    state: dict,
    reason: str,
    sentiment: str,
    actions_attempted: list[str],
    confidence: float,
) -> dict:
    """Build the structured handoff dict passed to the escalation agent."""
    order_id = state.get("order_id")
    relevant_order_info: dict = {}

    if order_id:
        try:
            raw_orders = _load_orders()
            relevant_order_info = raw_orders.get(order_id.upper(), {})
        except Exception as e:
            logger.warning("Could not load order data for handoff: %s", e)

    return {
        "issue_summary": reason,
        "actions_attempted": actions_attempted,
        "customer_sentiment": sentiment,
        "relevant_order_info": relevant_order_info,
        "escalation_reason": reason,
        "confidence_score": confidence,
        "agent": "nora_order_lookup",
    }


# ── Main node ─────────────────────────────────────────────────────────────────

def order_lookup_node(state: dict) -> dict:
    """
    LangGraph node — Order Lookup Agent (Nora).

    Reads:  messages, customer_id, session_id, order_id, metadata
    Writes: messages (appends AIMessage), agent_used, resolution_status,
            requires_escalation, retrieved_docs, retrieval_scores, metadata
            (metadata carries identity_verified, pending_otp, escalation_handoff,
            return_context — these are not top-level TypedDict fields so they live
            in metadata to survive LangGraph's state channel filtering)
    """
    messages = state.get("messages", [])
    customer_id = state.get("customer_id", "unknown")
    session_id = state.get("session_id", "unknown")
    order_id = state.get("order_id")
    metadata = dict(state.get("metadata", {}))  # shallow copy we'll extend

    past_context = metadata.get("past_context", "No prior interactions found.")
    identity_verified: bool = bool(metadata.get("identity_verified", False))

    last_human = next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "human"),
        "",
    )
    history_text = _get_history_text(messages)

    # ── Step 1: Identity verification ─────────────────────────────────────────
    if not identity_verified:
        if re.match(r"^\d{6}$", last_human.strip()):
            # Customer is submitting an OTP code
            pending_otp = metadata.get("pending_otp")
            if pending_otp and last_human.strip() == str(pending_otp):
                identity_verified = True
                metadata["identity_verified"] = True
                metadata["pending_otp"] = None
                logger.info("Identity verified via OTP for customer %s", customer_id)
                # Fall through to normal processing below
            else:
                logger.warning("Incorrect OTP attempt for customer %s", customer_id)
                return {
                    "messages": [AIMessage(content=(
                        "That code doesn't match what I have on file. "
                        "Please double-check and try again, or let me know "
                        "if you'd like me to resend a new code."
                    ))],
                    "agent_used": "order_lookup",
                    "resolution_status": "pending_verification",
                    "requires_escalation": False,
                    "retrieved_docs": [],
                    "retrieval_scores": [],
                    "metadata": metadata,
                }
        else:
            # First contact — dispatch OTP and wait
            try:
                otp, _masked = send_otp_email(customer_id)
                # _masked is intentionally discarded — never shown to customer
                metadata["pending_otp"] = otp
                metadata["identity_verified"] = False
                logger.info("OTP dispatched for customer %s", customer_id)
                return {
                    "messages": [AIMessage(content=(
                        "For your security, I've sent a 6-digit verification code "
                        "to your registered email address. "
                        "Please enter it here to continue."
                    ))],
                    "agent_used": "order_lookup",
                    "resolution_status": "pending_verification",
                    "requires_escalation": False,
                    "retrieved_docs": [],
                    "retrieval_scores": [],
                    "metadata": metadata,
                }
            except ValueError as e:
                logger.warning("OTP send failed — customer not found: %s", e)
                return {
                    "messages": [AIMessage(content=(
                        "I'm sorry, I wasn't able to locate your account. "
                        "Please contact our support team at support@shopease.com "
                        "or call 19123 and we'll sort this out right away."
                    ))],
                    "agent_used": "order_lookup",
                    "resolution_status": "needs_escalation",
                    "requires_escalation": True,
                    "retrieved_docs": [],
                    "retrieval_scores": [],
                    "metadata": metadata,
                }
            except RuntimeError as e:
                logger.error(
                    "OTP email delivery failed for customer %s: %s", customer_id, e
                )
                return {
                    "messages": [AIMessage(content=(
                        "I'm having trouble sending you a verification code right now. "
                        "Please try again in a moment, or contact us at "
                        "support@shopease.com or 19123."
                    ))],
                    "agent_used": "order_lookup",
                    "resolution_status": "needs_escalation",
                    "requires_escalation": True,
                    "retrieved_docs": [],
                    "retrieval_scores": [],
                    "metadata": metadata,
                }

    # ── Step 2: Emotion detection ─────────────────────────────────────────────
    emotion = _detect_emotion(last_human)

    if emotion in ("legal_threat", "extreme_anger"):
        logger.info(
            "Escalating due to emotion=%s for customer %s", emotion, customer_id
        )
        handoff = _build_handoff(
            state=state,
            reason=f"Customer expressed {emotion.replace('_', ' ')}",
            sentiment=emotion,
            actions_attempted=["identity_verification", "emotion_detection"],
            confidence=0.95,
        )
        metadata["escalation_handoff"] = handoff

        try:
            from memory.long_term import LongTermMemory
            LongTermMemory().save_interaction(
                customer_id=customer_id,
                session_id=session_id,
                summary=(
                    f"Escalated from order lookup due to {emotion.replace('_', ' ')}. "
                    f"Preview: {last_human[:120]}"
                ),
                metadata={
                    "intent": "escalation",
                    "reason": emotion,
                    "agent": "nora_order_lookup",
                },
            )
        except Exception as e:
            logger.warning("Could not save escalation to long-term memory: %s", e)

        return {
            "messages": [AIMessage(content=(
                "I understand this is extremely frustrating, and I sincerely apologise "
                "for the experience you've had. I'm connecting you right now with a "
                "senior specialist who has full authority to resolve this for you."
            ))],
            "agent_used": "order_lookup",
            "resolution_status": "escalated",
            "requires_escalation": True,
            "retrieved_docs": [],
            "retrieval_scores": [],
            "metadata": metadata,
        }

    # ── Step 3: Fetch order data ──────────────────────────────────────────────
    order_info = "No order information found."
    order_dict: Optional[dict] = None  # raw dict for modifications and handoffs

    if order_id:
        order_info = get_order_tool.invoke({"order_id": order_id})
        try:
            raw_orders = _load_orders()
            order_dict = raw_orders.get(order_id.strip().upper())
            if order_dict and order_dict.get("customer_id") != customer_id:
                logger.warning(
                    "Ownership violation: customer %s tried to view order %s "
                    "(belongs to %s)",
                    customer_id, order_id, order_dict.get("customer_id"),
                )
                return {
                    "messages": [AIMessage(content=(
                        "I'm sorry, that order doesn't appear in your account. "
                        "Please double-check the order ID and try again."
                    ))],
                    "agent_used": "order_lookup",
                    "resolution_status": "resolved",
                    "requires_escalation": False,
                    "retrieved_docs": [],
                    "retrieval_scores": [],
                    "metadata": metadata,
                }
        except Exception as e:
            logger.warning("Could not load raw order for ownership check: %s", e)
    else:
        order_info = list_customer_orders_tool.invoke({"customer_id": customer_id})

    # ── Step 4: RAG retrieval ─────────────────────────────────────────────────
    retrieved_docs: list[dict] = []
    retrieval_scores: list[float] = []
    kb_context = "No relevant articles found."
    low_confidence = False

    try:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.retrieve(query=last_human, top_k_final=settings.top_k_rerank)
        if docs:
            retrieved_docs = [
                {"content": d.content, "source": d.source, "score": d.rerank_score}
                for d in docs
            ]
            retrieval_scores = [d.rerank_score for d in docs]
            kb_context = retriever.format_for_prompt(docs)
            if all(s < 0.3 for s in retrieval_scores):
                low_confidence = True
                logger.debug(
                    "Low retrieval confidence for Nora query: %.60s", last_human
                )
    except Exception as e:
        logger.warning("RAG retrieval failed in order_lookup: %s", e)

    # ── Step 5: Modification request handling ────────────────────────────────
    modification_result: Optional[str] = None
    mod_intent = _detect_modification_intent(last_human)

    if mod_intent and order_id and order_dict:
        order_status = order_dict.get("status", "unknown")
        oid_upper = order_id.strip().upper()

        if order_status != "processing":
            in_motion = order_status in ("in_transit", "delayed", "shipped")
            modification_result = (
                f"I'm sorry, I can't modify this order — it is currently "
                f"'{order_status.replace('_', ' ').title()}'. "
                + (
                    "Once an order has shipped it can no longer be changed. "
                    "I can connect you with a specialist if you need further help."
                    if in_motion else
                    "Modifications are only possible while an order is still processing."
                )
            )
            logger.info(
                "Modification denied for order %s (status=%s, intent=%s)",
                oid_upper, order_status, mod_intent,
            )
        else:
            try:
                if mod_intent == "cancel":
                    modification_result = cancel_order_tool.invoke({
                        "order_id": oid_upper,
                        "customer_id": customer_id,
                    })

                elif mod_intent == "update_address":
                    addr_match = re.search(
                        r"(?:to|address[:\s]+)([\d][\w\s,.\-]{8,80})",
                        last_human, re.IGNORECASE,
                    )
                    if addr_match:
                        modification_result = update_address_tool.invoke({
                            "order_id": oid_upper,
                            "customer_id": customer_id,
                            "new_address": addr_match.group(1).strip(),
                        })
                    else:
                        modification_result = (
                            "Please provide your complete new shipping address "
                            "(street, city, postal code) and I'll update it right away."
                        )

                elif mod_intent == "update_quantity":
                    found_item = next(
                        (item for item in order_dict.get("items", [])
                         if item["name"].lower() in last_human.lower()),
                        None,
                    )
                    qty_match = re.search(r"\b(\d+)\b", last_human)
                    if found_item and qty_match:
                        modification_result = update_quantity_tool.invoke({
                            "order_id": oid_upper,
                            "customer_id": customer_id,
                            "product_id": found_item["product_id"],
                            "new_qty": int(qty_match.group(1)),
                        })
                    else:
                        modification_result = (
                            "Please tell me which item you'd like to update "
                            "and the new quantity you need."
                        )

                elif mod_intent == "remove_item":
                    found_item = next(
                        (item for item in order_dict.get("items", [])
                         if item["name"].lower() in last_human.lower()),
                        None,
                    )
                    if found_item:
                        modification_result = remove_item_tool.invoke({
                            "order_id": oid_upper,
                            "customer_id": customer_id,
                            "product_id": found_item["product_id"],
                        })
                    else:
                        modification_result = (
                            "Please specify which item you'd like to remove "
                            "and I'll take care of it immediately."
                        )

            except Exception as e:
                logger.warning("Order modification failed: %s", e)
                modification_result = (
                    "I wasn't able to complete that modification right now. "
                    "Please try again or contact us at support@shopease.com."
                )

    # ── Step 6: Return / exchange routing detection ────────────────────────────
    return_context_str = "No return request detected."
    resolution_status = "resolved"

    if _detect_return_intent(last_human) and order_dict:
        return_context = {
            "order_id": order_id,
            "purchase_date": order_dict.get("created_at"),
            "delivered_at": order_dict.get("delivered_at"),
            "items": order_dict.get("items", []),
            "customer_id": customer_id,
        }
        metadata["return_context"] = return_context
        return_context_str = (
            f"Return context collected for routing to Returns Specialist: "
            f"{json.dumps(return_context)}"
        )
        # On the next customer turn the supervisor will route to policy_returns
        resolution_status = "pending_routing"
        logger.info(
            "Return intent detected for order %s — context stored in metadata",
            order_id,
        )

    # ── Step 7: Confidence / escalation check ─────────────────────────────────
    requires_escalation = False
    escalation_handoff: Optional[dict] = None

    if low_confidence and not order_dict:
        logger.warning(
            "Low confidence and no order data — escalating for customer %s",
            customer_id,
        )
        requires_escalation = True
        escalation_handoff = _build_handoff(
            state=state,
            reason="Low confidence in available information — no matching order found",
            sentiment=emotion if emotion != "neutral" else "neutral",
            actions_attempted=["rag_retrieval"],
            confidence=0.2,
        )
        metadata["escalation_handoff"] = escalation_handoff

    # ── Step 8: Generate LLM response ─────────────────────────────────────────
    response = ""
    try:
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", NORA_SYSTEM),
            ("human", NORA_HUMAN),
        ])
        chain = prompt | llm | StrOutputParser()
        invoke_kwargs = {
            "customer_id": customer_id,
            "order_info": order_info,
            "kb_context": kb_context,
            "past_context": past_context,
            "modification_result": modification_result or "No modification requested.",
            "return_context": return_context_str,
            "history": history_text,
            "question": last_human,
        }
        response = chain.invoke(invoke_kwargs)

        # ── Step 9: Self-reflection ──────────────────────────────────────────
        if _needs_reflection(response):
            logger.debug("Nora: self-reflection triggered")
            try:
                response = _reflect(invoke_kwargs, initial_answer=response, llm=llm)
            except Exception as ref_e:
                logger.warning("Nora: reflection failed: %s", ref_e)

    except Exception as e:
        logger.error("Order lookup agent (Nora) LLM call failed: %s", e)
        response = (
            "I'm having trouble retrieving your order details right now. "
            "Please try again in a moment or contact us at support@shopease.com "
            "or call 19123."
        )

    # ── Step 10: Finalise escalation handoff ──────────────────────────────────
    if requires_escalation:
        if escalation_handoff is None:
            escalation_handoff = _build_handoff(
                state=state,
                reason="Escalation required",
                sentiment=emotion,
                actions_attempted=["order_lookup", "rag_retrieval"],
                confidence=0.5,
            )
            metadata["escalation_handoff"] = escalation_handoff

        try:
            from memory.long_term import LongTermMemory
            LongTermMemory().save_interaction(
                customer_id=customer_id,
                session_id=session_id,
                summary=(
                    f"Escalated from Nora (order lookup): "
                    f"{escalation_handoff.get('issue_summary', '')}"
                ),
                metadata={
                    "intent": "escalation",
                    "agent": "nora_order_lookup",
                },
            )
        except Exception as e:
            logger.warning("Could not save escalation to long-term memory: %s", e)

        resolution_status = "escalated"

    logger.info(
        "Order Lookup Agent (Nora) | customer=%s order_id=%s status=%s emotion=%s",
        customer_id, order_id, resolution_status, emotion,
    )

    return {
        "messages": [AIMessage(content=response)],
        "agent_used": "order_lookup",
        "resolution_status": resolution_status,
        "requires_escalation": requires_escalation,
        "retrieved_docs": retrieved_docs,
        "retrieval_scores": retrieval_scores,
        "metadata": metadata,
    }
