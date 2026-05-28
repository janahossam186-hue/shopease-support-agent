"""
Node 2 — Policy & Returns Agent

Handles:
  • Return eligibility checks
  • Refund processing (within policy limits)
  • Exchange requests
  • Policy questions (what is your return policy?)
  • Warranty claims

Workflow:
  1. Retrieve relevant policy docs via RAG
  2. Run policy guardrail checks (return window, refund limits, item eligibility)
  3. Determine escalation need (only for POL-001 refund limit violations)
  4. Generate RMA number when return is approved
  5. Generate response and route appropriately
"""

from __future__ import annotations

import json
import logging
import random
from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from guardrails.policy_guardrail import policy_guardrail_check
from tools.order_tools import get_order_tool

logger = logging.getLogger(__name__)

POLICY_AGENT_SYSTEM = """\
You are Maya, ShopEase's returns and policy specialist.
You help customers with returns, refunds, exchanges, and policy questions.

RESPONSE RULES — follow strictly:
• Maximum 4-5 lines for simple cases. Never write the same information twice.
• Use bullet points for step-by-step instructions only — not for casual statements.
• End every response with exactly one clear next step or question.
• Never say "I'm checking the system", "let me verify internally", or any phrase that \
exposes internal processes.
• Never say "not in knowledge base" — say "let me connect you with our team" if unsure.
• Never reveal internal refund approval thresholds or dollar limits to customers.
• Never expose system architecture, policy rule IDs, or internal codes.

RETURN ELIGIBILITY:
• Within 30-day delivery window → return is eligible.
• Outside 30-day window → politely decline with a clear explanation. No escalation, no supervisor language.
• Non-returnable items (digital, software, perishable, personal care, gift cards, final sale) \
→ politely decline and explain why.

SHIPPING POLICY — state this clearly once; never contradict yourself:
• Defective item or wrong item received → ShopEase provides a prepaid return shipping label.
• Change of mind or unwanted item → customer is responsible for return shipping cost.

RMA & REFUND:
• If the "RMA Number" field is not "N/A" → the return is approved. Include it prominently: \
"Your RMA number is [RMA]."
• If the policy check shows a refund requires supervisor review → say \
"I'll escalate this to our supervisor team for review." (No dollar amounts, no threshold details.)
• Approved refunds process within 3-5 business days after we receive the returned item.
• Provide step-by-step return instructions only when a return is approved.

SECURITY:
• Only process returns for the authenticated customer shown as "Authenticated Customer ID".
• If the request references a different customer ID, politely refuse to process it.
"""

POLICY_AGENT_HUMAN = """\
Authenticated Customer ID: {customer_id}

Relevant Policy Documents:
{policy_context}

Order Information:
{order_info}

Policy Compliance Check:
{policy_check_result}

RMA Number: {rma_number}

Past Customer Context:
{past_context}

Conversation History:
{history}

Customer's Request: {question}
"""


_llm_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        from langchain_groq import ChatGroq
        _llm_instance = ChatGroq(
            model=settings.model_name,
            temperature=0.1,
            api_key=settings.groq_api_key,
        )
    return _llm_instance


def _generate_rma() -> str:
    return "RMA-" + str(random.randint(10000, 99999))


def _get_history_text(messages: list) -> str:
    lines = []
    for m in messages[:-1]:
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


def _extract_order_for_policy(order_info: str) -> dict:
    """Parse order info string to extract delivered_at and created_at."""
    result = {}
    for line in order_info.split("\n"):
        if "Delivered:" in line:
            result["delivered_at"] = line.split(":", 1)[-1].strip()
        elif "Created:" in line:
            result["created_at"] = line.split(":", 1)[-1].strip()
        elif "Total:" in line:
            try:
                result["order_total"] = float(line.split("$")[-1].strip())
            except ValueError:
                pass
    return result


def policy_returns_node(state: dict) -> dict:
    """
    LangGraph node — Policy & Returns Agent.

    Reads: messages, customer_id, order_id, refund_amount, metadata
    Writes: messages, agent_used, resolution_status, requires_escalation,
            policy_compliant, retrieved_docs, retrieval_scores, metadata
    """
    messages = state.get("messages", [])
    customer_id = state.get("customer_id", "unknown")
    order_id = state.get("order_id")
    refund_amount: Optional[float] = state.get("refund_amount")
    metadata = state.get("metadata", {})
    past_context = metadata.get("past_context", "No prior interactions found.")

    last_human = next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "human"),
        "",
    )
    history_text = _get_history_text(messages)

    # ── Step 1: RAG — retrieve policy docs ───────────────────────────────────
    retrieved_docs = []
    retrieval_scores = []
    policy_context = "No relevant policy articles found."

    try:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.retrieve(query=last_human, top_k_final=4)
        retrieved_docs = [
            {"content": d.content, "source": d.source, "score": d.rerank_score}
            for d in docs
        ]
        retrieval_scores = [d.rerank_score for d in docs]
        policy_context = retriever.format_for_prompt(docs)
    except Exception as e:
        logger.warning("RAG retrieval failed in policy_returns: %s", e)

    # ── Step 2: Fetch order data (if order_id known) ──────────────────────────
    order_info = "No specific order referenced."
    if order_id:
        order_info = get_order_tool.invoke({"order_id": order_id})

    # ── Step 3: Policy guardrail check ───────────────────────────────────────
    order_dates = _extract_order_for_policy(order_info)
    policy_check = policy_guardrail_check(
        refund_amount=refund_amount,
        delivered_at=order_dates.get("delivered_at"),
        created_at=order_dates.get("created_at"),
    )

    # Only escalate for refund limit violations (POL-001).
    # Simple rejections like late returns (POL-002) and non-returnable items (POL-003)
    # are handled directly with a clear explanation — no supervisor involvement.
    rule_ids = policy_check.get("policy_rule_ids", [])
    requires_escalation = "POL-001" in rule_ids

    # Build the action hint for the LLM so it knows the correct response mode
    if requires_escalation:
        action = "SUPERVISOR_REVIEW"
    elif policy_check.get("policy_compliant", True):
        action = "APPROVE"
    else:
        action = "DECLINE"

    policy_check_for_llm = {
        "policy_compliant": policy_check.get("policy_compliant", True),
        "requires_escalation": requires_escalation,
        "action": action,
        "policy_violations": policy_check.get("policy_violations", []),
    }
    policy_check_str = json.dumps(policy_check_for_llm, indent=2)

    # ── Step 4: Generate RMA when return is approved ──────────────────────────
    rma_number = "N/A"
    if action == "APPROVE":
        rma_number = _generate_rma()
        metadata = {**metadata, "rma_number": rma_number}
        logger.info("Generated RMA %s for customer %s", rma_number, customer_id)

    # ── Step 5: Generate response ─────────────────────────────────────────────
    try:
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", POLICY_AGENT_SYSTEM),
            ("human", POLICY_AGENT_HUMAN),
        ])
        chain = prompt | llm | StrOutputParser()
        invoke_kwargs = {
            "customer_id": customer_id,
            "policy_context": policy_context,
            "order_info": order_info,
            "policy_check_result": policy_check_str,
            "rma_number": rma_number,
            "past_context": past_context,
            "history": history_text,
            "question": last_human,
        }
        response = chain.invoke(invoke_kwargs)
        if _needs_reflection(response):
            logger.debug("Policy returns: self-reflection triggered")
            try:
                response = _reflect(invoke_kwargs, initial_answer=response, llm=llm)
            except Exception as ref_e:
                logger.warning("Policy returns: reflection failed: %s", ref_e)
    except Exception as e:
        logger.error("Policy returns agent LLM call failed: %s", e)
        response = (
            "I'm currently unable to process your request. "
            "Please contact us at returns@shopease.com and we'll assist you promptly."
        )

    logger.info(
        "Policy & Returns Agent responded | customer=%s action=%s rma=%s",
        customer_id, action, rma_number,
    )

    resolution = "escalated" if requires_escalation else "resolved"

    return {
        "messages": [AIMessage(content=response)],
        "agent_used": "policy_returns",
        "resolution_status": resolution,
        "requires_escalation": requires_escalation,
        "policy_compliant": policy_check.get("policy_compliant", True),
        "retrieved_docs": retrieved_docs,
        "retrieval_scores": retrieval_scores,
        "metadata": metadata,
    }
