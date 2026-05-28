"""
Evaluation Metrics — SQLite-backed interaction logger.

Tracked per-interaction:
  • Latency (ms)
  • Intent classification
  • Agent used
  • Resolution status (resolved / escalated / blocked)
  • Guardrail pass/fail
  • RAG: number of retrieved docs + average relevance score
  • Policy compliance
  • Output toxicity score

Tracked aggregates (computed on-demand by the dashboard):
  • Resolution rate = resolved / total
  • Policy compliance rate
  • Average retrieval quality (avg rerank score)
  • P50 / P90 / P99 latency
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

_DB_PATH = None
_conn: sqlite3.Connection | None = None

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT    NOT NULL,
    customer_id         TEXT    NOT NULL,
    timestamp           TEXT    NOT NULL,
    intent              TEXT,
    agent_used          TEXT,
    resolution_status   TEXT,
    latency_ms          REAL,
    guardrail_passed    INTEGER DEFAULT 1,
    retrieved_doc_count INTEGER DEFAULT 0,
    avg_retrieval_score REAL    DEFAULT 0.0,
    policy_compliant    INTEGER DEFAULT 1,
    toxicity_score      REAL    DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_customer   ON interactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_timestamp  ON interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_session    ON interactions(session_id);
"""


def _get_conn() -> sqlite3.Connection:
    global _conn, _DB_PATH
    db_path = Path(settings.eval_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if _conn is None or _DB_PATH != str(db_path):
        _conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _conn.executescript(CREATE_TABLE_SQL)
        _conn.commit()
        _DB_PATH = str(db_path)

    return _conn


def init_db() -> None:
    """Explicitly initialise the database (useful in scripts)."""
    _get_conn()
    logger.info("Evaluation database initialised at %s", settings.eval_db_path)


def log_interaction(
    session_id: str,
    customer_id: str,
    intent: str,
    agent_used: str,
    resolution_status: str,
    latency_ms: float,
    guardrail_passed: bool = True,
    retrieved_doc_count: int = 0,
    avg_retrieval_score: float = 0.0,
    policy_compliant: bool = True,
    toxicity_score: float = 0.0,
) -> None:
    """Insert one interaction record into the evaluation database."""
    try:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO interactions (
                session_id, customer_id, timestamp, intent, agent_used,
                resolution_status, latency_ms, guardrail_passed,
                retrieved_doc_count, avg_retrieval_score,
                policy_compliant, toxicity_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                customer_id,
                datetime.utcnow().isoformat(),
                intent,
                agent_used,
                resolution_status,
                round(latency_ms, 2),
                int(guardrail_passed),
                retrieved_doc_count,
                round(avg_retrieval_score, 4),
                int(policy_compliant),
                round(toxicity_score, 4),
            ),
        )
        conn.commit()
    except Exception as e:
        logger.warning("Failed to log interaction metric: %s", e)


def get_metrics_df(hours: int = 24) -> pd.DataFrame:
    """Return recent interactions as a pandas DataFrame."""
    try:
        conn = _get_conn()
        query = """
            SELECT * FROM interactions
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn, params=[f"-{hours} hours"])
        # Fix boolean columns
        for col in ["guardrail_passed", "policy_compliant"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        return df
    except Exception as e:
        logger.warning("Failed to read metrics: %s", e)
        return pd.DataFrame()


def compute_summary(df: pd.DataFrame) -> dict:
    """Compute aggregate KPIs from a metrics DataFrame."""
    if df.empty:
        return {}

    total = len(df)
    return {
        "total_interactions": total,
        "resolution_rate": df["resolution_status"].eq("resolved").mean(),
        "escalation_rate": df["resolution_status"].eq("escalated").mean(),
        "blocked_rate": df["resolution_status"].eq("blocked").mean(),
        "guardrail_block_rate": (~df["guardrail_passed"]).mean(),
        "policy_compliance_rate": df["policy_compliant"].mean(),
        "avg_latency_ms": df["latency_ms"].mean(),
        "p50_latency_ms": df["latency_ms"].quantile(0.5),
        "p90_latency_ms": df["latency_ms"].quantile(0.9),
        "avg_retrieval_score": df["avg_retrieval_score"].mean(),
        "avg_docs_retrieved": df["retrieved_doc_count"].mean(),
        "avg_toxicity_score": df["toxicity_score"].mean(),
        "intent_distribution": df["intent"].value_counts().to_dict(),
        "agent_distribution": df["agent_used"].value_counts().to_dict(),
    }
