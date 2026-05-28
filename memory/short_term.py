"""
Short-term memory: per-session conversation history managed by LangGraph's
MemorySaver checkpointer.  Each session is isolated by its thread_id.

LangGraph automatically saves/restores the full AgentState between turns
when a checkpointer is attached to the compiled graph.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_checkpointer = None


def get_checkpointer():
    """
    Return a singleton LangGraph checkpointer.
    Uses SqliteSaver when available (persists across process restarts),
    falls back to in-memory MemorySaver otherwise.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    db_path = Path(settings.memory_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
        logger.info("Short-term memory: SqliteSaver at %s", db_path)
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
        logger.info("Short-term memory: in-memory MemorySaver (install langgraph[sqlite] for persistence)")

    return _checkpointer


def get_session_config(session_id: str) -> dict:
    """
    Build the LangGraph run config that scopes checkpointing to a session.

    Usage::
        config = get_session_config("session-abc123")
        graph.invoke(state, config=config)
    """
    return {"configurable": {"thread_id": session_id}}
