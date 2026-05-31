"""
Long-term memory: stores customer interaction summaries in ChromaDB so the
agent can recall past context across different sessions.

Each memory entry captures:
- customer_id
- session_id
- A summary of what happened (intent, resolution, key facts)
- Embedding for semantic search

This lets the agent say things like "Last time you had an issue with ORD-10001,
we processed a replacement…" even when it's a new session.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from rag.embeddings import LocalEmbeddings

logger = logging.getLogger(__name__)

LONG_TERM_COLLECTION = "customer_memory"


class LongTermMemory:
    """
    Semantic memory store backed by ChromaDB.

    Usage::

        mem = LongTermMemory()
        # Save a session summary
        mem.save_interaction(
            customer_id="CUST-001",
            session_id="sess-abc",
            summary="Customer asked about ORD-10001, which was delayed. Offered $10 credit.",
            metadata={"intent": "order_lookup", "resolution": "resolved"},
        )
        # Recall relevant past context
        past = mem.recall(customer_id="CUST-001", query="delayed order")
    """

    def __init__(self):
        self.embedder = LocalEmbeddings(model_name=settings.embedding_model)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=LONG_TERM_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def save_interaction(
        self,
        customer_id: str,
        session_id: str,
        summary: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Embed and store a session summary for a customer."""
        ts = datetime.utcnow().isoformat()
        doc_id = f"mem_{customer_id}_{session_id}_{int(time.time() * 1000)}"

        meta = {
            "customer_id": customer_id,
            "session_id": session_id,
            "timestamp": ts,
        }
        if metadata:
            # ChromaDB metadata values must be str/int/float/bool
            for k, v in metadata.items():
                meta[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v

        embedding = self.embedder.embed_query(summary)

        self.collection.upsert(
            ids=[doc_id],
            documents=[summary],
            embeddings=[embedding],
            metadatas=[meta],
        )
        logger.debug("Long-term memory saved for customer %s (session %s)", customer_id, session_id)

    def recall(
        self,
        customer_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve the most relevant past interactions for a customer.

        Returns a list of dicts with 'content', 'timestamp', 'session_id'.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)

        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where={"customer_id": customer_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning("Long-term memory recall failed: %s", e)
            return []

        memories = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            relevance = 1.0 - dist
            if relevance > 0.3:  # only return reasonably relevant memories
                memories.append(
                    {
                        "content": doc,
                        "timestamp": meta.get("timestamp", ""),
                        "session_id": meta.get("session_id", ""),
                        "relevance": round(relevance, 3),
                        "metadata": meta,
                    }
                )

        return memories

    def format_for_prompt(self, memories: list[dict]) -> str:
        """Format recalled memories for injection into an agent prompt."""
        if not memories:
            return "No relevant past interactions found for this customer."

        lines = ["**Past interactions with this customer:**"]
        for mem in memories:
            ts = mem.get("timestamp", "")[:10]  # date only
            lines.append(f"• [{ts}] {mem['content']}")
        return "\n".join(lines)

    def get_customer_history_count(self, customer_id: str) -> int:
        """Return total stored memories for a customer."""
        try:
            result = self.collection.get(where={"customer_id": customer_id})
            return len(result["ids"])
        except Exception:
            return 0
