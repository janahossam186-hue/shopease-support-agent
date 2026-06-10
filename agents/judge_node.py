"""
LLM Judge Node — evaluates the agent's last response on three RAG quality dimensions.

Scores (0.0–1.0):
  faithfulness       — response is grounded in the retrieved docs (no hallucination)
  answer_relevancy   — response directly answers the customer's question
  context_precision  — retrieved docs are relevant to the customer's question

Runs after finalize_metrics; saves scores to eval_metrics.db and writes them
back into state["metadata"]["judge_scores"] for downstream use.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from evaluation.metrics import log_llm_scores

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = """\
You are an objective evaluation judge for a customer-support AI system.
Score the agent's response on three dimensions, each from 0.00 to 1.00.

FAITHFULNESS (0–1)
  Does every factual claim in the response appear in the context documents?
  0 = significant hallucination; 1 = fully grounded in context.

ANSWER_RELEVANCY (0–1)
  Does the response directly and completely address the customer's question?
  0 = off-topic or completely unhelpful; 1 = precise and complete answer.

CONTEXT_PRECISION (0–1)
  Are the retrieved context documents useful for answering this question?
  0 = all documents are irrelevant; 1 = every document is highly relevant.

Rules:
- If no context documents were retrieved, set faithfulness and context_precision to 0.
- Be objective; do not reward vague or padded responses.
- Respond ONLY with valid JSON on a single line — no explanation, no markdown.\
"""

_JUDGE_HUMAN = """\
CUSTOMER QUESTION:
{question}

RETRIEVED CONTEXT DOCUMENTS:
{context}

AGENT RESPONSE:
{response}

Return ONLY this JSON (replace the zeros with your scores):
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}}\
"""


def _get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.model_name,
        temperature=0.0,
        api_key=settings.groq_api_key,
    )


def _extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of an LLM reply."""
    match = re.search(r'\{[^{}]+\}', text)
    if match:
        return json.loads(match.group())
    return json.loads(text.strip())


def judge_node(state: dict) -> dict:
    messages      = state.get("messages", [])
    retrieved_docs = state.get("retrieved_docs", [])
    session_id    = state.get("session_id", "unknown")
    customer_id   = state.get("customer_id", "unknown")
    metadata      = state.get("metadata", {})

    # Extract last human question and last AI response
    question = next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "human"),
        "",
    )
    response = next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "ai"),
        "",
    )

    # Nothing to score — guardrail blocked or no response generated
    if not question or not response:
        return {"metadata": {**metadata, "judge_scores": {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}}}

    # Build context text from retrieved docs (truncated to ~1 200 chars)
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
        context_parts.append(f"[{i}] {content[:400]}")
    context = "\n".join(context_parts) if context_parts else "No documents retrieved."

    scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}
    try:
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", _JUDGE_SYSTEM),
            ("human", _JUDGE_HUMAN),
        ])
        raw = (prompt | llm).invoke({
            "question": question[:600],
            "context":  context,
            "response": response[:600],
        }).content.strip()

        parsed = _extract_json(raw)
        for key in scores:
            val = parsed.get(key, 0.0)
            scores[key] = round(max(0.0, min(1.0, float(val))), 4)

        logger.info(
            "Judge scores | session=%s faithfulness=%.2f relevancy=%.2f precision=%.2f",
            session_id, scores["faithfulness"], scores["answer_relevancy"], scores["context_precision"],
        )
    except Exception as e:
        logger.warning("Judge node failed: %s", e)

    log_llm_scores(
        session_id=session_id,
        customer_id=customer_id,
        question=question,
        response=response,
        **scores,
    )

    return {"metadata": {**metadata, "judge_scores": scores}}
