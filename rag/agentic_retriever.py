"""
Agentic Retriever — wraps HybridRetriever with LLM-driven decisions.
Pipeline: decide → retrieve → grade → (retry once if needed)
"""
from __future__ import annotations

import json
import logging
import re

from config.settings import settings
from rag.retriever import RetrievedDoc, get_retriever

logger = logging.getLogger(__name__)

_instance = None


def get_agentic_retriever() -> "AgenticRetriever":
    global _instance
    if _instance is None:
        _instance = AgenticRetriever()
    return _instance


class AgenticRetriever:

    def __init__(self):
        self._retriever = get_retriever()
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(model=settings.model_name, temperature=0.0,
                                  api_key=settings.groq_api_key)
        return self._llm

    def _parse_json(self, text: str) -> dict:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            return json.loads(m.group()) if m else {}
        except json.JSONDecodeError:
            return {}

    def _decide(self, question: str) -> dict:
        prompt = f"""You are a retrieval decision agent for ShopEase Egypt's customer support system.

ShopEase knowledge base contains:
  product_catalog    — product prices, specs, features, stock availability,
                       warranty. Use for "what is the price of X", "is X
                       in stock", "what are the specs of X".

  product_manuals    — troubleshooting, error messages, burning smells,
                       device not working, setup guides, how-to steps,
                       cleaning, maintenance, strange noises, overheating.
                       USE THIS for ANY question where a product is
                       malfunctioning, behaving unexpectedly, or the
                       customer needs help using it.

  cosmetics_catalog  — skincare, beauty products, ingredients, how to use,
                       skin type recommendations.

  returns_policy     — return windows, refund rules, non-returnable items,
                       exchange policy, warranty claims.

  shipping_policy    — delivery options, timelines, shipping costs,
                       tracking, international shipping.

  store_info         — branch locations, contact info, payment methods,
                       promotions, operating hours, website help.

  faq                — common questions about orders, accounts, tracking,
                       payments, cancellations.

  recommendations    — trending products, bundles, seasonal offers,
                       gift ideas, bestsellers.

Think through this step by step before deciding:

STEP 1 — SHOULD WE RETRIEVE?
Is this question related to ShopEase products, policies, store info, or services?
  YES → continue to Step 2
  NO  → no retrieval needed (e.g. math question, unrelated topic, simple greeting)

STEP 2 — WHAT SHOULD WE SEARCH FOR?
What is the customer really asking about?
Identify the core topic and pick the best matching collection from the list above.
Leave collection_hint empty to search across all collections.

COLLECTION SELECTION RULE:
- If the customer mentions a product AND a problem/issue/error/smell/
  sound/malfunction → always pick product_manuals
- If the customer asks about price, specs, availability → product_catalog
- If the customer mentions returning or refunding → returns_policy
- When in doubt between product_catalog and product_manuals,
  pick product_manuals — troubleshooting is more useful than specs

STEP 3 — DOES THE QUERY NEED REWRITING?
Is the customer's message vague, implicit, or indirect?
  Vague:    "it keeps burning"          → "appliance burn warning error troubleshooting"
  Implicit: "I want my money back"      → "refund request return policy"
  Indirect: "something smells in box"  → "damaged product return complaint"
If the question is already clear and specific, keep it as-is.

Now respond with a JSON object only — no extra text outside the JSON:
{{
  "should_retrieve": true or false,
  "reason": "one sentence explaining why",
  "collection_hint": "one collection name from the list above, or empty string to search all",
  "search_query": "the optimised search query (or empty string if no retrieval)"
}}

Customer question: {question}"""

        try:
            raw = self._get_llm().invoke(prompt).content.strip()
            return self._parse_json(raw)
        except Exception as e:
            logger.warning("Decision LLM failed: %s — defaulting to retrieve", e)
            return {"should_retrieve": True, "search_query": question, "collection_hint": ""}

    def _grade(self, question: str, docs: list[RetrievedDoc]) -> bool:
        doc_texts = "\n\n".join(f"[{i+1}] {d.content[:200]}" for i, d in enumerate(docs[:3]))
        prompt = f"""You are checking whether retrieved documents are relevant to a customer question.

Customer question: {question}

Retrieved documents:
{doc_texts}

Are these documents relevant and useful for answering the customer's question?
Answer with YES or NO only."""

        try:
            return self._get_llm().invoke(prompt).content.strip().upper().startswith("YES")
        except Exception as e:
            logger.warning("Grade LLM failed: %s — treating as passed", e)
            return True

    def _retry_query(self, question: str, failed_query: str) -> str:
        prompt = f"""You are improving a failed search query for ShopEase Egypt's knowledge base.
The previous search did not return useful results.

STEP 1 — UNDERSTAND THE INTENT
What is the customer actually trying to do or find out?
(e.g. "it stopped working" → customer has a broken product, wants help or a return)

STEP 2 — WRITE A BETTER QUERY
Use different keywords that are more likely to match knowledge base articles.
Try synonyms, more specific terms, or related topics you haven't tried yet.

Customer question: {question}
Failed query: {failed_query}

Return ONLY the new search query — no explanation, no JSON, no preamble."""

        try:
            return self._get_llm().invoke(prompt).content.strip()
        except Exception as e:
            logger.warning("Retry LLM failed: %s — using original question", e)
            return question

    def _run(self, question: str, top_k: int) -> tuple[list[RetrievedDoc], bool]:
        """Core pipeline. Returns (docs, was_skipped).
        was_skipped=True means the LLM decided retrieval was unnecessary."""
        # Step 1 — decide whether to retrieve and how
        decision = self._decide(question)
        if not decision.get("should_retrieve", True):
            return [], True

        search_query    = decision.get("search_query") or question
        collection_hint = decision.get("collection_hint") or None

        # Step 2 — retrieve
        docs = self._retriever.retrieve(
            query=search_query,
            top_k_final=top_k,
            source_filter=collection_hint,
        )

        # Step 3 — skip grade if cross-encoder is already confident
        # Cross-encoder scores are raw logits — can be negative; use top score with logit threshold
        top_score = max((d.rerank_score for d in docs), default=0.0)
        if top_score >= 1.0:
            logger.debug("Grade skipped — top rerank score %.3f >= 1.0", top_score)
            return docs, False

        # Step 4 — grade + retry once with widened search if docs are poor or empty
        if not docs or not self._grade(question, docs):
            new_query = self._retry_query(question, search_query)
            docs = self._retriever.retrieve(
                query=new_query,
                top_k_final=top_k,
                source_filter=None,
            )

        return docs, False

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedDoc]:
        docs, _ = self._run(question, top_k)
        return docs

    def retrieve_with_skip(self, question: str, top_k: int = 3) -> tuple[list[RetrievedDoc], bool]:
        """Like retrieve(), but also returns whether retrieval was intentionally skipped."""
        return self._run(question, top_k)
