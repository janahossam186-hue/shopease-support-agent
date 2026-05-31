"""
LangChain tools that wrap the HybridRetriever for policy and FAQ queries.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_retriever():
    """Lazily load and cache the retriever (builds BM25 index on first call)."""
    from rag.retriever import HybridRetriever
    r = HybridRetriever()
    r.build_bm25_index()
    return r


@tool
def search_policy_tool(query: str) -> str:
    """
    Search the ShopEase shipping and returns policy knowledge base.
    Use this for questions about return windows, refund timelines,
    shipping costs, tracking, and other policy-related topics.
    """
    retriever = _get_retriever()
    docs = retriever.retrieve(query, source_filter=None, top_k_final=3)
    # Filter to policy docs only
    policy_docs = [d for d in docs if "policy" in d.source]
    if not policy_docs:
        policy_docs = docs  # fall back to all
    return retriever.format_for_prompt(policy_docs)


@tool
def search_faq_tool(query: str) -> str:
    """
    Search the ShopEase FAQ database for common customer questions.
    Use this for general support questions about orders, payments,
    shipping, returns, account management, and product information.
    """
    retriever = _get_retriever()
    docs = retriever.retrieve(query, source_filter=None, top_k_final=3)
    faq_docs = [d for d in docs if d.source == "faq"]
    if not faq_docs:
        faq_docs = docs
    return retriever.format_for_prompt(faq_docs)


@tool
def search_all_knowledge_tool(query: str) -> str:
    """
    Search the full ShopEase knowledge base (products, FAQs, and policies).
    Use this for broad questions that might span multiple knowledge sources.
    """
    retriever = _get_retriever()
    docs = retriever.retrieve(query)
    return retriever.format_for_prompt(docs)
