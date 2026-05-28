"""
General Agent — ShopEase Customer Support

Handles everything that isn't a specific order lookup, returns/refund request,
or escalation:
  • Friendly greeting and small talk
  • Product usage guidance (from product manuals)
  • Skincare / beauty advice and recommendations
  • Cross-selling and product bundle suggestions
  • Store locations, promotions, website help
  • Account / payment issues
  • Trending products and seasonal offers
  • Personalised recommendations based on customer history

This agent is designed to feel warm, human, and genuinely helpful —
not like a chatbot reading from a script.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

GENERAL_AGENT_SYSTEM = """\
You are Layla, ShopEase's friendly customer support specialist.
You work for ShopEase Egypt — a leading e-commerce platform selling electronics, appliances,
beauty products, and more.

Your personality:
- Warm, friendly, and genuinely enthusiastic about helping
- You speak naturally — like a real person, not a corporate robot
- You're strictly honest — you never invent, guess, or extrapolate facts

════════════════════════════════════════════════════════
STRICT GROUNDING RULE — read carefully and never violate:
════════════════════════════════════════════════════════
Your answers MUST be grounded EXCLUSIVELY in the "Relevant Knowledge Base Articles"
provided in the user message.

• If the knowledge base articles contain the answer → answer from them directly.
  Quote or paraphrase the source; do not add information from your own training data.

• If the knowledge base section says "NO_DOCS_FOUND" or contains no article relevant
  to the customer's question → respond with a warm, honest message such as:
  "I'm sorry, I don't have that information in our knowledge base right now.
   For the most accurate answer, please contact our support team at support@shopease.com
   or call 19123 — they'll be happy to help! 😊"

• NEVER make up product names, ingredients, prices, steps, addresses, promo codes,
  store hours, policy details, or any other factual claim not present in the articles.

• The only exception to the grounding rule is pure greetings and small talk
  (e.g., "Hi", "How are you?") — for those you may respond warmly without citing articles.
════════════════════════════════════════════════════════

Keep responses natural and conversational. Use bullet points or numbered lists only
when walking through steps or listing recommendations — not for casual chat.
End factual responses with a gentle offer to help further: "Is there anything else I can help you with? 😊"
"""

GENERAL_AGENT_HUMAN = """\
Customer ID: {customer_id}

Past Customer Context (long-term memory):
{past_context}

Relevant Knowledge Base Articles:
[Knowledge base status: {kb_status}]
{kb_context}

REMINDER: Answer ONLY from the knowledge base articles above.
If the knowledge base status is NO_DOCS_FOUND or the articles do not address the
customer's question, say you don't have that information and direct them to support.

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
            temperature=0.2,  # low temp for strict RAG grounding
            api_key=settings.groq_api_key,
        )
    return _llm_instance


def _get_history_text(messages: list) -> str:
    """Format recent conversation history for the prompt."""
    lines = []
    for m in messages[:-1]:  # exclude the latest human message
        role = getattr(m, "type", "unknown")
        if role == "human":
            lines.append(f"Customer: {m.content}")
        elif role == "ai":
            lines.append(f"Layla: {m.content}")
    return "\n".join(lines) if lines else "This is the start of the conversation."


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


_NO_DOCS_CONTEXT = (
    "NO_DOCS_FOUND — the knowledge base returned no articles relevant to this query."
)

def _retrieve_knowledge(query: str, customer_id: str) -> tuple[str, str, list[dict], list[float]]:
    """
    Run hybrid RAG retrieval. Falls back gracefully if ChromaDB is empty or unavailable.
    Returns (kb_status, formatted_context, retrieved_docs, scores).
    """
    try:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.retrieve(query=query, top_k_final=settings.top_k_rerank)
        if not docs:
            logger.info("RAG returned 0 docs for query: %.60s", query)
            return "NO_DOCS_FOUND", _NO_DOCS_CONTEXT, [], []
        retrieved_docs = [
            {"content": d.content, "source": d.source, "score": d.rerank_score}
            for d in docs
        ]
        retrieval_scores = [d.rerank_score for d in docs]
        kb_context = retriever.format_for_prompt(docs)
        return "DOCS_FOUND", kb_context, retrieved_docs, retrieval_scores
    except Exception as e:
        logger.warning("RAG retrieval failed in general_agent: %s", e)
        return "NO_DOCS_FOUND", _NO_DOCS_CONTEXT, [], []


def general_agent_node(state: dict) -> dict:
    """
    LangGraph node — General Agent (Layla).

    Reads:  messages, customer_id, metadata
    Writes: messages (appends AIMessage), agent_used, resolution_status,
            retrieved_docs, retrieval_scores
    """
    messages = state.get("messages", [])
    customer_id = state.get("customer_id", "unknown")
    metadata = state.get("metadata", {})
    past_context = metadata.get("past_context", "No previous interactions on record.")

    # Last customer message
    last_human = next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "human"),
        "",
    )
    history_text = _get_history_text(messages)

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    kb_status, kb_context, retrieved_docs, retrieval_scores = _retrieve_knowledge(
        last_human, customer_id
    )
    logger.debug("General agent RAG status=%s docs=%d", kb_status, len(retrieved_docs))

    # ── Generate response ─────────────────────────────────────────────────────
    try:
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", GENERAL_AGENT_SYSTEM),
            ("human", GENERAL_AGENT_HUMAN),
        ])
        chain = prompt | llm | StrOutputParser()
        invoke_kwargs = {
            "customer_id": customer_id,
            "past_context": past_context,
            "kb_status": kb_status,
            "kb_context": kb_context,
            "history": history_text,
            "question": last_human,
        }
        response = chain.invoke(invoke_kwargs)
        if _needs_reflection(response):
            logger.debug("General agent: self-reflection triggered")
            try:
                response = _reflect(invoke_kwargs, initial_answer=response, llm=llm)
            except Exception as ref_e:
                logger.warning("General agent: reflection failed: %s", ref_e)
    except Exception as e:
        logger.error("General agent LLM call failed: %s", e)
        response = (
            "Hi there! 😊 Thanks for reaching out to ShopEase. "
            "I'm having a small technical hiccup right now, but I'll be right back with you. "
            "In the meantime, you can reach us on WhatsApp at +20 100 123 4567 or call 19123. "
            "We're here for you!"
        )

    logger.info(
        "General Agent (Layla) responded | customer=%s query_preview=%.60s",
        customer_id, last_human,
    )

    return {
        "messages": [AIMessage(content=response)],
        "agent_used": "general",
        "resolution_status": "resolved",
        "retrieved_docs": retrieved_docs,
        "retrieval_scores": retrieval_scores,
    }
