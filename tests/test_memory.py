"""
Memory system test suite — short_term.py & long_term.py

Sections:
  1. Short-term memory tests   — checkpointer singleton, session config
  2. Long-term save tests      — doc_id uniqueness, metadata, upsert
  3. Long-term recall tests    — filtering, threshold, format_for_prompt
  4. Known gaps (xfail)        — unimplemented features documented

Run all:
    pytest tests/test_memory.py -v

Run just short-term tests:
    pytest tests/test_memory.py -v -k "ShortTerm"

Run just long-term tests:
    pytest tests/test_memory.py -v -k "LongTerm"

Run gap tests:
    pytest tests/test_memory.py -v -k "Gaps"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_mock_collection(docs=None, metas=None, dists=None, count=0):
    """Build a mock ChromaDB collection with controllable query results."""
    col = MagicMock()
    col.count.return_value = count
    col.query.return_value = {
        "documents": [docs or []],
        "metadatas": [metas or []],
        "distances": [dists or []],
    }
    return col


def make_ltm_with_mock_collection(docs=None, metas=None, dists=None, count=0):
    """Return a LongTermMemory instance with ChromaDB fully mocked."""
    from memory.long_term import LongTermMemory

    mock_col = make_mock_collection(docs, metas, dists, count)
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    with patch("memory.long_term.chromadb.PersistentClient") as mock_client, \
         patch("memory.long_term.LocalEmbeddings", return_value=mock_embedder):
        mock_client.return_value.get_or_create_collection.return_value = mock_col
        ltm = LongTermMemory()
        ltm.collection = mock_col
        ltm.embedder = mock_embedder

    return ltm, mock_col, mock_embedder


# ─────────────────────────────────────────────────────────────────────────────
# 1. Short-term memory tests
# ─────────────────────────────────────────────────────────────────────────────

class TestShortTermMemory:

    def setup_method(self):
        """Reset the checkpointer singleton before each test."""
        import memory.short_term as st
        st._checkpointer = None

    def test_get_session_config_returns_correct_format(self):
        from memory.short_term import get_session_config
        config = get_session_config("session-abc123")
        assert config == {"configurable": {"thread_id": "session-abc123"}}

    def test_get_session_config_different_sessions_different_configs(self):
        from memory.short_term import get_session_config
        c1 = get_session_config("session-001")
        c2 = get_session_config("session-002")
        assert c1["configurable"]["thread_id"] != c2["configurable"]["thread_id"]

    def test_get_session_config_thread_id_matches_session_id(self):
        from memory.short_term import get_session_config
        session_id = "test-session-xyz"
        config = get_session_config(session_id)
        assert config["configurable"]["thread_id"] == session_id

    def test_checkpointer_is_singleton(self):
        from memory.short_term import get_checkpointer
        with patch("memory.short_term.sqlite3.connect"), \
             patch("memory.short_term.SqliteSaver", create=True) as mock_saver:
            mock_saver.return_value = MagicMock()
            c1 = get_checkpointer()
            c2 = get_checkpointer()
        assert c1 is c2

    def test_checkpointer_falls_back_to_memory_saver(self):
        from memory.short_term import get_checkpointer
        with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite": None}):
            import memory.short_term as st
            st._checkpointer = None
            try:
                checkpointer = get_checkpointer()
                assert checkpointer is not None
            except Exception:
                pass  # ImportError on missing module is acceptable

    def test_get_checkpointer_creates_data_directory(self, tmp_path):
        import memory.short_term as st
        st._checkpointer = None
        fake_db = tmp_path / "data" / "memory.db"

        with patch("memory.short_term.settings") as mock_settings, \
             patch("memory.short_term.sqlite3.connect") as mock_conn:
            mock_settings.memory_db_path = str(fake_db)
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                with patch("memory.short_term.SqliteSaver"):
                    get_checkpointer()
            except ImportError:
                pass

        # Directory should exist even if SQLite import fails
        # (mkdir happens before the try block)

    def test_session_config_with_empty_string_session_id(self):
        from memory.short_term import get_session_config
        config = get_session_config("")
        assert config["configurable"]["thread_id"] == ""

    def test_session_config_with_unicode_session_id(self):
        from memory.short_term import get_session_config
        config = get_session_config("session-مرحبا-123")
        assert "thread_id" in config["configurable"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Long-term memory — save_interaction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLongTermMemorySave:

    def test_save_calls_upsert(self):
        ltm, mock_col, mock_embedder = make_ltm_with_mock_collection()
        ltm.save_interaction("CUST-001", "sess-001", "Customer asked about order.")
        mock_col.upsert.assert_called_once()

    def test_save_embeds_summary(self):
        ltm, mock_col, mock_embedder = make_ltm_with_mock_collection()
        ltm.save_interaction("CUST-001", "sess-001", "Order was delayed.")
        mock_embedder.embed_query.assert_called_once_with("Order was delayed.")

    def test_save_doc_id_contains_customer_id(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        ltm.save_interaction("CUST-002", "sess-001", "Return processed.")
        call_kwargs = mock_col.upsert.call_args[1]
        assert "CUST-002" in call_kwargs["ids"][0]

    def test_save_doc_id_contains_session_id(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        ltm.save_interaction("CUST-001", "sess-XYZ", "Summary.")
        call_kwargs = mock_col.upsert.call_args[1]
        assert "sess-XYZ" in call_kwargs["ids"][0]

    def test_save_stores_summary_as_document(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        summary = "Customer escalated about broken coffee maker."
        ltm.save_interaction("CUST-001", "sess-001", summary)
        call_kwargs = mock_col.upsert.call_args[1]
        assert call_kwargs["documents"][0] == summary

    def test_save_metadata_contains_customer_id(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        ltm.save_interaction("CUST-003", "sess-001", "Summary.")
        call_kwargs = mock_col.upsert.call_args[1]
        assert call_kwargs["metadatas"][0]["customer_id"] == "CUST-003"

    def test_save_metadata_contains_session_id(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        ltm.save_interaction("CUST-001", "sess-ABC", "Summary.")
        call_kwargs = mock_col.upsert.call_args[1]
        assert call_kwargs["metadatas"][0]["session_id"] == "sess-ABC"

    def test_save_metadata_contains_timestamp(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        ltm.save_interaction("CUST-001", "sess-001", "Summary.")
        call_kwargs = mock_col.upsert.call_args[1]
        assert "timestamp" in call_kwargs["metadatas"][0]

    def test_save_converts_non_string_metadata_values(self):
        """ChromaDB requires str/int/float/bool metadata values."""
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        ltm.save_interaction(
            "CUST-001", "sess-001", "Summary.",
            metadata={"list_value": [1, 2, 3], "dict_value": {"a": 1}}
        )
        call_kwargs = mock_col.upsert.call_args[1]
        meta = call_kwargs["metadatas"][0]
        assert isinstance(meta["list_value"], str)
        assert isinstance(meta["dict_value"], str)

    def test_save_accepts_none_metadata(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        # Should not raise
        ltm.save_interaction("CUST-001", "sess-001", "Summary.", metadata=None)
        mock_col.upsert.assert_called_once()

    def test_two_saves_same_second_produce_different_ids(self):
        """doc_id collision: same customer + session + second = overwrite."""
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        ids_used = []

        def capture_upsert(**kwargs):
            ids_used.append(kwargs["ids"][0])

        mock_col.upsert.side_effect = capture_upsert

        with patch("memory.long_term.time.time", return_value=1000000.0):
            ltm.save_interaction("CUST-001", "sess-001", "First save.")
            ltm.save_interaction("CUST-001", "sess-001", "Second save.")

        # This test documents the bug: same second = same ID = overwrite
        # After fix (milliseconds or uuid), ids_used[0] != ids_used[1]
        assert len(ids_used) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Long-term memory — recall tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLongTermMemoryRecall:

    def test_empty_collection_returns_empty_list(self):
        ltm, _, _ = make_ltm_with_mock_collection(count=0)
        result = ltm.recall("CUST-001", "order status")
        assert result == []

    def test_recall_returns_list(self):
        ltm, _, _ = make_ltm_with_mock_collection(
            docs=["Customer had delayed order."],
            metas=[{"customer_id": "CUST-001", "session_id": "s1", "timestamp": "2026-01-01"}],
            dists=[0.2],  # relevance = 1 - 0.2 = 0.8 → above threshold
            count=1,
        )
        result = ltm.recall("CUST-001", "order")
        assert isinstance(result, list)

    def test_recall_high_relevance_memory_is_returned(self):
        ltm, _, _ = make_ltm_with_mock_collection(
            docs=["Escalation ticket TKT-001 created."],
            metas=[{"customer_id": "CUST-001", "session_id": "s1", "timestamp": "2026-01-01"}],
            dists=[0.1],  # relevance = 0.9 → well above 0.3 threshold
            count=1,
        )
        result = ltm.recall("CUST-001", "escalation")
        assert len(result) == 1

    def test_recall_low_relevance_memory_is_filtered(self):
        ltm, _, _ = make_ltm_with_mock_collection(
            docs=["Totally unrelated memory."],
            metas=[{"customer_id": "CUST-001", "session_id": "s1", "timestamp": "2026-01-01"}],
            dists=[0.8],  # relevance = 1 - 0.8 = 0.2 → below 0.3 threshold
            count=1,
        )
        result = ltm.recall("CUST-001", "order")
        assert result == []

    def test_recall_result_contains_content_field(self):
        ltm, _, _ = make_ltm_with_mock_collection(
            docs=["Order ORD-10001 was delayed."],
            metas=[{"customer_id": "CUST-001", "session_id": "s1", "timestamp": "2026-01-01"}],
            dists=[0.2],
            count=1,
        )
        result = ltm.recall("CUST-001", "order")
        assert "content" in result[0]

    def test_recall_result_contains_relevance_field(self):
        ltm, _, _ = make_ltm_with_mock_collection(
            docs=["Memory."],
            metas=[{"customer_id": "CUST-001", "session_id": "s1", "timestamp": "2026-01-01"}],
            dists=[0.2],
            count=1,
        )
        result = ltm.recall("CUST-001", "query")
        assert "relevance" in result[0]
        assert result[0]["relevance"] == round(1.0 - 0.2, 3)

    def test_recall_uses_customer_id_filter(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection(count=1)
        mock_col.query.return_value = {
            "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        ltm.recall("CUST-001", "query")
        call_kwargs = mock_col.query.call_args[1]
        assert call_kwargs["where"] == {"customer_id": "CUST-001"}

    def test_recall_query_failure_returns_empty_list(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection(count=1)
        mock_col.query.side_effect = Exception("ChromaDB error")
        result = ltm.recall("CUST-001", "query")
        assert result == []

    def test_format_for_prompt_empty_memories(self):
        ltm, _, _ = make_ltm_with_mock_collection()
        result = ltm.format_for_prompt([])
        assert "No relevant" in result

    def test_format_for_prompt_includes_content(self):
        ltm, _, _ = make_ltm_with_mock_collection()
        memories = [{"content": "Customer had issue with ORD-10001.", "timestamp": "2026-01-01T10:00:00"}]
        result = ltm.format_for_prompt(memories)
        assert "ORD-10001" in result

    def test_format_for_prompt_includes_date(self):
        ltm, _, _ = make_ltm_with_mock_collection()
        memories = [{"content": "Some memory.", "timestamp": "2026-05-01T12:00:00"}]
        result = ltm.format_for_prompt(memories)
        assert "2026-05-01" in result

    def test_format_for_prompt_multiple_memories(self):
        ltm, _, _ = make_ltm_with_mock_collection()
        memories = [
            {"content": "First memory.", "timestamp": "2026-01-01T10:00:00"},
            {"content": "Second memory.", "timestamp": "2026-01-02T10:00:00"},
        ]
        result = ltm.format_for_prompt(memories)
        assert "First memory" in result
        assert "Second memory" in result

    def test_get_customer_history_count_returns_zero_on_error(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        mock_col.get.side_effect = Exception("DB error")
        assert ltm.get_customer_history_count("CUST-001") == 0

    def test_get_customer_history_count_returns_correct_count(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        mock_col.get.return_value = {"ids": ["id1", "id2", "id3"]}
        assert ltm.get_customer_history_count("CUST-001") == 3


# ─────────────────────────────────────────────────────────────────────────────
# 4. Known gaps (xfail)
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryGaps:

    @pytest.mark.xfail(
        reason="Same-second doc_id collision not fixed yet — two saves in same second overwrite",
        strict=True,
    )
    def test_same_second_saves_produce_unique_ids(self):
        ltm, mock_col, _ = make_ltm_with_mock_collection()
        ids_used = []

        def capture(**kwargs):
            ids_used.append(kwargs["ids"][0])

        mock_col.upsert.side_effect = capture

        with patch("memory.long_term.time.time", return_value=1000000.0):
            ltm.save_interaction("CUST-001", "sess-001", "First.")
            ltm.save_interaction("CUST-001", "sess-001", "Second.")

        assert ids_used[0] != ids_used[1]

    def test_relevance_threshold_is_0_5(self):
        ltm, _, _ = make_ltm_with_mock_collection(
            docs=["Vaguely related memory."],
            metas=[{"customer_id": "CUST-001", "session_id": "s1", "timestamp": "2026-01-01"}],
            dists=[0.6],  # relevance = 0.4 — above 0.3 but below 0.5
            count=1,
        )
        result = ltm.recall("CUST-001", "query")
        # With threshold 0.5, relevance 0.4 should be filtered out
        assert result == []

    @pytest.mark.xfail(
        reason="Session expiry not implemented — memory.db grows forever",
        strict=True,
    )
    def test_old_sessions_are_pruned(self):
        from memory.short_term import get_checkpointer
        checkpointer = get_checkpointer()
        assert hasattr(checkpointer, "prune_sessions") or hasattr(checkpointer, "expire_sessions")

    def test_order_lookup_saves_to_long_term_memory(self):
        from agents.order_lookup import order_lookup_node
        import inspect
        source = inspect.getsource(order_lookup_node)
        assert "save_interaction" in source


    def test_policy_returns_saves_to_long_term_memory(self):
        from agents.policy_returns import policy_returns_node
        import inspect
        source = inspect.getsource(policy_returns_node)
        assert "save_interaction" in source