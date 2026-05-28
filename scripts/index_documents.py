"""
Document Indexing Script
Run once (or after updating data files) to populate ChromaDB.

    python scripts/index_documents.py
"""

import sys
import logging
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("indexer")


def main():
    from config.settings import settings
    settings.apply_langsmith_env()
    settings.ensure_dirs()

    from rag.indexer import DocumentIndexer

    logger.info("=" * 60)
    logger.info("ShopEase Document Indexer")
    logger.info("=" * 60)

    indexer = DocumentIndexer()

    # Show current state before indexing
    before = indexer.get_collection_stats()
    logger.info("Collections before indexing: %s", before)

    # Run the indexing pipeline
    indexer.index_all()

    # Show state after
    after = indexer.get_collection_stats()
    logger.info("Collections after indexing: %s", after)

    # Verify BM25 index can be built
    logger.info("Verifying BM25 index construction…")
    from rag.retriever import HybridRetriever
    retriever = HybridRetriever()
    retriever.build_bm25_index()

    # Run a test query
    test_query = "How do I return a damaged product?"
    logger.info("Test query: '%s'", test_query)
    docs = retriever.retrieve(test_query, top_k_final=2)
    for i, doc in enumerate(docs, 1):
        logger.info(
            "  [%d] source=%s score=%.3f preview='%s…'",
            i, doc.source, doc.rerank_score, doc.content[:80],
        )

    logger.info("Indexing complete ✓")


if __name__ == "__main__":
    main()
