"""
Step-Back Retriever — BM25 + dense retrieval on a step-back transformed query.
Step-back prompting rewrites a specific customer question into a broader, more
general one that better matches knowledge base documents.
Used only for evaluation. Not used in production agents.
"""
from __future__ import annotations

import logging

from config.settings import settings
from rag.retriever import get_retriever, _RRF_K

logger = logging.getLogger(__name__)

_instance = None


def get_stepback_retriever() -> "StepBackRetriever":
    global _instance
    if _instance is None:
        _instance = StepBackRetriever()
    return _instance


class StepBackRetriever:
    """
    Pipeline:
        1. LLM broadens the query (step-back)
        2. BM25 + dense search on the broader query
        3. RRF fusion
        4. Return top-k  (no cross-encoder reranking)
    """

    def __init__(self):
        self._r   = get_retriever()   # reuse BM25 + dense from HybridRetriever
        self._llm = None              # lazy-loaded on first retrieve call

    def _get_llm(self):
        if self._llm is None:
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                model=settings.model_name,
                temperature=0.0,
                api_key=settings.groq_api_key,
            )
        return self._llm

    def _stepback(self, query: str) -> str:
        """Return a broader, more general version of the query."""
        prompt = (
            "You are improving search queries for ShopEase Egypt "
            "(electronics, appliances, beauty products).\n"
            "Rewrite the customer question below as a broader, more general question "
            "that will better match knowledge base articles about products, policies, or store info.\n"
            "Return ONLY the broader question — no explanation.\n\n"
            f"Customer question: {query}\n"
            "Broader question:"
        )
        try:
            result = self._get_llm().invoke(prompt).content.strip()
            logger.debug("Step-back: %r → %r", query[:50], result[:50])
            return result
        except Exception as e:
            logger.warning("Step-back LLM failed, using original query: %s", e)
            return query   # fallback to original

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Step-back the query, then BM25 + dense + RRF, return top-k."""
        stepback_query = self._stepback(query)
        k_ret = max(top_k * 3, 10)

        bm25_docs  = self._r._bm25_search(stepback_query, k_ret)
        dense_docs = self._r._dense_search(stepback_query, k_ret)

        # RRF fusion
        merged = {}
        for rank, doc in enumerate(bm25_docs, 1):
            doc.hybrid_score = 1.0 / (_RRF_K + rank)
            merged[doc.doc_id] = doc
        for rank, doc in enumerate(dense_docs, 1):
            rrf = 1.0 / (_RRF_K + rank)
            if doc.doc_id in merged:
                merged[doc.doc_id].hybrid_score += rrf
            else:
                doc.hybrid_score = rrf
                merged[doc.doc_id] = doc

        results = sorted(merged.values(), key=lambda d: d.hybrid_score, reverse=True)[:top_k]
        for doc in results:
            doc.rerank_score = doc.hybrid_score
        return results
