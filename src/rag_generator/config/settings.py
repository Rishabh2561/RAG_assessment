"""Typed application configuration.

All environment-specific values live here. Values are read from ``RAG_*`` environment
variables and an optional ``.env`` file; the Anthropic key uses the SDK's standard
``ANTHROPIC_API_KEY`` name. Nothing else in the codebase should read the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["dense", "bm25", "hybrid"]


class Settings(BaseSettings):
    """Validated configuration for every component."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    data_dir: Path = Path(".rag_data")
    default_collection: str = Field("default", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    max_file_mb: float = Field(50, gt=0)

    # Chunking
    chunk_size: int = Field(700, ge=100, le=8000)
    chunk_overlap: int = Field(140, ge=0)
    min_chunk_chars: int = Field(20, ge=1)

    # Embeddings
    embedding_provider: Literal["fastembed", "hashing"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = Field(32, ge=1)
    hashing_dimensions: int = Field(1024, ge=64)
    model_cache_dir: Path | None = None

    # Storage backend
    vector_store: Literal["numpy"] = "numpy"

    # Retrieval
    retrieval_mode: RetrievalMode = "dense"  # chosen by evaluation; see DECISION_LOG D8
    top_k: int = Field(5, ge=1, le=50)
    candidate_pool: int = Field(20, ge=1, le=200)
    rrf_k: int = Field(60, ge=1)
    # Calibrated for bge-small-en-v1.5 (DECISION_LOG D9); recalibrate if the model changes.
    min_relevance: float = Field(0.50, ge=0.0, le=1.0)

    # Reranking
    reranker: Literal["none", "cross_encoder"] = "none"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Generation
    llm_provider: Literal["anthropic", "none"] = "anthropic"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "low"
    llm_max_tokens: int = Field(4096, ge=256)
    llm_timeout_s: float = Field(60, gt=0)
    llm_max_retries: int = Field(2, ge=0, le=10)
    llm_temperature: float | None = Field(None, ge=0.0, le=1.0)
    anthropic_server_fallback: bool = True
    anthropic_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "RAG_ANTHROPIC_API_KEY")
    )
    max_context_chars: int = Field(12000, ge=1000)
    max_question_chars: int = Field(2000, ge=10)

    # Agentic corrective retry (ADR-009)
    query_rewrite: bool = False
    max_rewrites: int = Field(3, ge=1, le=5)

    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "text"
    log_content: bool = False

    @model_validator(mode="after")
    def _check_cross_field_constraints(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"RAG_CHUNK_SIZE ({self.chunk_size})"
            )
        if self.top_k > self.candidate_pool:
            raise ValueError(
                f"RAG_TOP_K ({self.top_k}) must not exceed RAG_CANDIDATE_POOL "
                f"({self.candidate_pool})"
            )
        return self

    @property
    def collections_dir(self) -> Path:
        return self.data_dir / "collections"
