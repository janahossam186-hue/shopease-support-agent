"""
Centralised configuration loaded from .env (or environment variables).
Access via:  from config.settings import settings
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    model_name: str = Field(default="llama-3.3-70b-versatile", alias="MODEL_NAME")
    temperature: float = Field(default=0.1, alias="TEMPERATURE")

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: str = Field(default="true", alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(
        default="ecommerce-support-agent", alias="LANGCHAIN_PROJECT"
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT"
    )

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./data/chroma_db", alias="CHROMA_PERSIST_DIR"
    )

    # ── HuggingFace ───────────────────────────────────────────────────────────
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # ── RAG ──────────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    top_k_retrieval: int = Field(default=10, alias="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=3, alias="TOP_K_RERANK")
    bm25_weight: float = Field(default=0.4, alias="BM25_WEIGHT")
    dense_weight: float = Field(default=0.6, alias="DENSE_WEIGHT")

    # ── Guardrails ────────────────────────────────────────────────────────────
    max_refund_amount: float = Field(default=500.0, alias="MAX_REFUND_AMOUNT")
    return_window_days: int = Field(default=30, alias="RETURN_WINDOW_DAYS")
    toxicity_threshold: float = Field(default=0.6, alias="TOXICITY_THRESHOLD")

    # ── Memory / Eval paths ───────────────────────────────────────────────────
    memory_db_path: str = Field(default="./data/memory.db", alias="MEMORY_DB_PATH")
    eval_db_path: str = Field(default="./data/eval_metrics.db", alias="EVAL_DB_PATH")

    # ── Email (Gmail SMTP) ────────────────────────────────────────────────
    gmail_address: str = Field(
        default="shopease.support.eg@gmail.com", alias="GMAIL_ADDRESS"
    )
    gmail_app_password: str = Field(
        default="", alias="GMAIL_APP_PASSWORD"
    )
    report_recipient: str = Field(
        default="", alias="REPORT_RECIPIENT"
    )

    def apply_langsmith_env(self) -> None:
        """Push LangSmith variables into os.environ so the SDK picks them up."""
        os.environ.setdefault("LANGCHAIN_API_KEY", self.langchain_api_key)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", self.langchain_tracing_v2)
        os.environ.setdefault("LANGCHAIN_PROJECT", self.langchain_project)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", self.langchain_endpoint)
        if self.groq_api_key:
            os.environ.setdefault("GROQ_API_KEY", self.groq_api_key)
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", self.hf_token)

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        for path_str in [self.chroma_persist_dir, "./data/logs"]:
            Path(path_str).mkdir(parents=True, exist_ok=True)
        for db_path in [self.memory_db_path, self.eval_db_path]:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
