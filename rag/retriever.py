"""
HybridRetriever — combines BM25 sparse retrieval with ChromaDB dense retrieval,
then re-ranks with a cross-encoder.

Pipeline:
  1. BM25 + dense search run in parallel (ThreadPoolExecutor, max_workers=2)
  2. Reciprocal Rank Fusion (RRF) merges results — no weight tuning needed
  3. Hard cap at MAX_RERANK_CANDIDATES before cross-encoder
  4. Skip-rerank fast path when top RRF score is clearly dominant
  5. Cross-encoder reranking of final candidates

Singletons (module-level, built once per process):
  - _chroma_client_instance  — one persistent ChromaDB connection
  - _retriever_instance      — one HybridRetriever with BM25 already built
  Call get_retriever() from agents; never instantiate HybridRetriever directly.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from config.settings import settings
from rag.embeddings import LocalEmbeddings, CrossEncoderReranker
from rag.indexer import COLLECTION_ALL

logger = logging.getLogger(__name__)

# Hard cap on cross-encoder input to keep reranking latency bounded (~110ms/doc)
MAX_RERANK_CANDIDATES = 20

# RRF smoothing constant (standard value; higher k → less sensitive to rank differences)
_RRF_K = 60


@dataclass
class RetrievedDoc:
    """A single retrieved document with its provenance and scores."""
    doc_id: str
    content: str
    source: str
    metadata: dict
    bm25_score: float = 0.0
    dense_score: float = 0.0
    hybrid_score: float = 0.0   # RRF fusion score
    rerank_score: float = 0.0


# ── Module-level singletons ───────────────────────────────────────────────────

_chroma_client_instance: Optional[chromadb.PersistentClient] = None
_retriever_instance: Optional["HybridRetriever"] = None


def _get_chroma_client() -> chromadb.PersistentClient:
    """Return the process-level ChromaDB client, creating it on first call."""
    global _chroma_client_instance
    if _chroma_client_instance is None:
        _chroma_client_instance = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client opened at %s", settings.chroma_persist_dir)
    return _chroma_client_instance


def get_retriever() -> "HybridRetriever":
    """
    Return the process-level HybridRetriever with BM25 index already built.

    Agents call this instead of instantiating HybridRetriever directly.
    The BM25 index is built once and reused for all requests.

    Prints a clear error to stderr if the ChromaDB index is empty (indexer
    has not been run yet) — retrieval will silently return [] in that case.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
        _retriever_instance.build_bm25_index()
        if _retriever_instance._bm25 is None:
            print(
                "\n⚠  ChromaDB index is empty — no documents loaded.\n"
                "  Run:  .venv\\Scripts\\python.exe scripts/index_documents.py\n"
                "  before starting the app.\n",
                file=sys.stderr,
            )
            logger.error(
                "ChromaDB index is empty. Run scripts/index_documents.py first."
            )
    return _retriever_instance


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid search: parallel BM25 + dense → RRF fusion → cross-encoder reranking.

    Use get_retriever() to obtain the singleton instance; never call
    build_bm25_index() manually — get_retriever() handles that.
    """

    def __init__(self):
        self.embedder = LocalEmbeddings(model_name=settings.embedding_model)
        self.reranker = CrossEncoderReranker(model_name=settings.reranker_model)
        self.chroma_client = _get_chroma_client()

        # BM25 index (built once from ChromaDB documents)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_metas: list[dict] = []

    # ── BM25 index construction ───────────────────────────────────────────────

    def build_bm25_index(self) -> None:
        """Load all docs from ChromaDB and build the BM25 index in memory."""
        try:
            collection = self.chroma_client.get_collection(COLLECTION_ALL)
        except Exception:
            logger.warning(
                "ChromaDB collection '%s' not found. Run the indexer first.", COLLECTION_ALL
            )
            return

        total = collection.count()
        if total == 0:
            logger.warning("Collection is empty. Run the indexer first.")
            return

        # Fetch all documents in batches (ChromaDB has a page-size limit)
        batch_size = 500
        all_docs, all_ids, all_metas = [], [], []
        for offset in range(0, total, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            all_docs.extend(result["documents"])
            all_ids.extend(result["ids"])
            all_metas.extend(result["metadatas"])

        self._bm25_docs = all_docs
        self._bm25_ids = all_ids
        self._bm25_metas = all_metas

        tokenised = [doc.lower().split() for doc in all_docs]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("BM25 index built with %d documents.", len(all_docs))

    # ── internal search methods ───────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedDoc]:
        if self._bm25 is None:
            logger.warning("BM25 index not built. Call build_bm25_index() first.")
            return []

        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                meta = self._bm25_metas[idx] or {}
                results.append(
                    RetrievedDoc(
                        doc_id=self._bm25_ids[idx],
                        content=self._bm25_docs[idx],
                        source=meta.get("source", "unknown"),
                        metadata=meta,
                        bm25_score=float(scores[idx]),
                    )
                )
        return results

    def _dense_search(self, query: str, top_k: int) -> list[RetrievedDoc]:
        try:
            collection = self.chroma_client.get_collection(COLLECTION_ALL)
        except Exception:
            logger.warning("ChromaDB collection not found.")
            return []

        query_embedding = self.embedder.embed_query(query)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            cosine_sim = 1.0 - dist
            meta = meta or {}
            results.append(
                RetrievedDoc(
                    doc_id=result["ids"][0][len(results)],
                    content=doc,
                    source=meta.get("source", "unknown"),
                    metadata=meta,
                    dense_score=max(0.0, cosine_sim),
                )
            )
        return results

    @staticmethod
    def _should_skip_rerank(rrf_scores: list[float], threshold: float = 0.1) -> bool:
        """Return True when the top RRF score gap is large enough to skip reranking."""
        if len(rrf_scores) < 2:
            return True
        return (rrf_scores[0] - rrf_scores[1]) > threshold

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k_retrieval: Optional[int] = None,
        top_k_final: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> list[RetrievedDoc]:
        """
        Full hybrid retrieval pipeline.

        1. BM25 + dense search run in parallel via ThreadPoolExecutor.
        2. Reciprocal Rank Fusion (RRF) merges by rank position — no weight tuning.
        3. Hard cap at MAX_RERANK_CANDIDATES before cross-encoder.
        4. Skip-rerank fast path when top RRF score gap exceeds threshold.
        5. Cross-encoder reranking for final ranking.

        Returns:
            List of RetrievedDoc sorted by rerank_score (or hybrid_score if skipped).
        """
        k_ret = top_k_retrieval or settings.top_k_retrieval
        k_fin = top_k_final or settings.top_k_rerank

        # 1. Parallel BM25 + dense retrieval
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self._bm25_search, query, k_ret)
            dense_future = executor.submit(self._dense_search, query, k_ret)
            bm25_results = bm25_future.result()
            dense_results = dense_future.result()

        # 2. Reciprocal Rank Fusion — rank = 1-indexed position in each sorted list
        merged: dict[str, RetrievedDoc] = {}

        for rank, doc in enumerate(bm25_results, 1):
            doc.hybrid_score = 1.0 / (_RRF_K + rank)
            merged[doc.doc_id] = doc

        for rank, doc in enumerate(dense_results, 1):
            rrf_score = 1.0 / (_RRF_K + rank)
            if doc.doc_id in merged:
                merged[doc.doc_id].hybrid_score += rrf_score
                merged[doc.doc_id].dense_score = doc.dense_score
            else:
                doc.hybrid_score = rrf_score
                merged[doc.doc_id] = doc

        # 3. Optional source filter + sort by RRF score
        candidates = list(merged.values())
        if source_filter:
            candidates = [d for d in candidates if d.source == source_filter] or candidates
        candidates.sort(key=lambda d: d.hybrid_score, reverse=True)

        # Hard cap before cross-encoder
        candidates = candidates[:min(len(candidates), MAX_RERANK_CANDIDATES)]

        if not candidates:
            return []

        # 4. Skip-rerank fast path
        rrf_scores = [d.hybrid_score for d in candidates]
        if self._should_skip_rerank(rrf_scores):
            gap = rrf_scores[0] - (rrf_scores[1] if len(rrf_scores) > 1 else 0.0)
            logger.debug(
                "Skip-rerank (RRF gap=%.4f > 0.1); returning top %d by RRF score", gap, k_fin
            )
            for doc in candidates[:k_fin]:
                doc.rerank_score = doc.hybrid_score
            return candidates[:k_fin]

        # 5. Cross-encoder reranking
        doc_texts = [d.content for d in candidates]
        reranked = self.reranker.rerank(query, doc_texts, top_k=k_fin)

        final_docs = []
        for orig_idx, rerank_score in reranked:
            doc = candidates[orig_idx]
            doc.rerank_score = rerank_score
            final_docs.append(doc)

        logger.debug(
            "Hybrid retrieval: %d BM25, %d dense → %d candidates → %d reranked",
            len(bm25_results), len(dense_results), len(candidates), len(final_docs),
        )
        return final_docs

    def format_for_prompt(self, docs: list[RetrievedDoc]) -> str:
        """Format retrieved documents into a clean string for LLM prompts."""
        if not docs:
            return "No relevant knowledge base articles found."

        parts = []
        for i, doc in enumerate(docs, 1):
            source_label = doc.source.replace("_", " ").title()
            parts.append(
                f"[{i}] Source: {source_label} (relevance: {doc.rerank_score:.2f})\n"
                f"{doc.content}"
            )
        return "\n\n---\n\n".join(parts)
