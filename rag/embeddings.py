"""
Singleton wrapper around a SentenceTransformer bi-encoder.
The model is downloaded from HuggingFace Hub on first use (~90 MB for MiniLM)
and cached to HF_HOME (defaults to ./data/hf_cache).
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Use HF_HOME if set; otherwise fall back to a local cache directory.
_HF_CACHE = os.environ.get("HF_HOME", "./data/hf_cache")

# Module-level singleton — loaded once per process.
_model = None


def _get_model(model_name: str):
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s (cache: %s)", model_name, _HF_CACHE)
            _model = SentenceTransformer(model_name, cache_folder=_HF_CACHE)
            logger.info("Embedding model loaded successfully.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            )
    return _model


class LocalEmbeddings:
    """
    LangChain-compatible embedding class using sentence-transformers.

    Usage::

        emb = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
        vectors = emb.embed_documents(["text 1", "text 2"])
        query_vec = emb.embed_query("my question")
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = _get_model(self.model_name)
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = _get_model(self.model_name)
        embedding = model.encode([text], show_progress_bar=False, normalize_embeddings=True)
        return embedding[0].tolist()


# Module-level singleton for the cross-encoder — loaded once per process.
_cross_encoder = None


class CrossEncoderReranker:
    """
    Reranks (query, doc) pairs using a cross-encoder.
    Downloads model on first call (~120 MB for ms-marco-MiniLM-L-6-v2)
    and caches to HF_HOME.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def _load(self):
        global _cross_encoder
        if _cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(
                    "Loading cross-encoder reranker: %s (cache: %s)",
                    self.model_name, _HF_CACHE,
                )
                # cache_dir must be passed via model_kwargs to avoid deprecation warning
                _cross_encoder = CrossEncoder(
                    self.model_name,
                    model_kwargs={"cache_dir": _HF_CACHE},
                )
                logger.info("Cross-encoder loaded successfully.")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. pip install sentence-transformers"
                )
        return _cross_encoder

    def rerank(
        self, query: str, documents: List[str], top_k: int = 3
    ) -> List[tuple[int, float]]:
        """
        Returns a list of (original_index, score) tuples sorted by descending score.
        """
        cross_encoder = self._load()
        pairs = [[query, doc] for doc in documents]
        scores: np.ndarray = cross_encoder.predict(pairs)
        indexed_scores = sorted(
            enumerate(scores.tolist()), key=lambda x: x[1], reverse=True
        )
        return indexed_scores[:top_k]
