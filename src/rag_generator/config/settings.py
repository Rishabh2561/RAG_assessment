"""Typed application configuration.

All environment-specific values live here. Values are read from ``RAG_*`` environment
variables and an optional ``.env`` file; API keys use their SDKs' standard names
(``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``). Nothing else in the codebase should read
the environment.

Model names and the relevance threshold default per provider when left unset; the
resolution happens once, at validation time, so every reader sees concrete values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, PrivateAttr, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["dense", "bm25", "hybrid"]

DEFAULT_LLM_MODELS = {"anthropic": "claude-opus-5-5", "openai": "gpt-5.5"}
LLMProviderName = Literal["anthropic", "openai"]
DEFAULT_EMBEDDING_MODELS = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
    "hashing": "hashing",
}
# Relevance-gate thresholds measured with `rag eval` (DECISION_LOG D9). Any other
# embedding model gets 0.0 (gate off) until calibrated: similarity scales differ by
# model, and an uncalibrated threshold can silently reject answerable questions.
CALIBRATED_MIN_RELEVANCE = {("fastembed", "BAAI/bge-small-en-v1.5"): 0.50}


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
    max_upload_files: int = Field(20, ge=1, le=1000)

    # Chunking
    chunk_size: int = Field(700, ge=100, le=8000)
    chunk_overlap: int = Field(140, ge=0)
    min_chunk_chars: int = Field(20, ge=1)

    # Embeddings
    embedding_provider: Literal["fastembed", "openai", "hashing"] = "fastembed"
    embedding_model: str | None = None  # None -> DEFAULT_EMBEDDING_MODELS[provider]
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
    # None -> CALIBRATED_MIN_RELEVANCE for the embedding model, else 0.0 (gate off).
    min_relevance: float | None = Field(None, ge=0.0, le=1.0)

    # Reranking
    reranker: Literal["none", "cross_encoder"] = "none"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Generation
    # auto -> whichever API key is configured (Anthropic first), else none (retrieval only)
    llm_provider: Literal["auto", "anthropic", "openai", "none"] = "auto"
    llm_model: str | None = None  # None -> DEFAULT_LLM_MODELS[provider]
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "low"  # Anthropic
    llm_max_tokens: int = Field(4096, ge=256)
    llm_timeout_s: float = Field(60, gt=0)
    llm_max_retries: int = Field(2, ge=0, le=10)
    llm_temperature: float | None = Field(None, ge=0.0, le=1.0)
    anthropic_server_fallback: bool = True
    anthropic_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "RAG_ANTHROPIC_API_KEY")
    )
    # OpenAI (used when RAG_LLM_PROVIDER=openai and/or RAG_EMBEDDING_PROVIDER=openai)
    openai_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("OPENAI_API_KEY", "RAG_OPENAI_API_KEY")
    )
    openai_base_url: str | None = None  # Azure / OpenAI-compatible servers (e.g. Ollama)
    # Only for reasoning models; sent only when set (non-reasoning models reject it).
    openai_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = (
        None
    )
    # Let API clients (e.g. the web UI) supply their own provider + key per request.
    allow_client_llm_keys: bool = True
    max_context_chars: int = Field(12000, ge=1000)
    max_question_chars: int = Field(2000, ge=10)

    # Agentic corrective retry (ADR-009)
    query_rewrite: bool = False
    max_rewrites: int = Field(3, ge=1, le=5)

    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "text"
    log_content: bool = False

    _gate_uncalibrated: bool = PrivateAttr(default=False)
    _llm_provider_auto: bool = PrivateAttr(default=False)

    @field_validator("anthropic_api_key", "openai_api_key", mode="after")
    @classmethod
    def _blank_key_is_unset(cls, value: SecretStr | None) -> SecretStr | None:
        """Treat blank values, and ``KEY=  # comment`` lines (which dotenv parses as the
        comment text), as no key, so a malformed .env can't select a provider."""
        if value is None:
            return None
        secret = value.get_secret_value().strip()
        return SecretStr(secret) if secret and not secret.startswith("#") else None

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

    @model_validator(mode="after")
    def _resolve_provider_defaults(self) -> Settings:
        if self.llm_provider == "auto":
            self.llm_provider = _provider_from_keys(self.anthropic_api_key, self.openai_api_key)
            self._llm_provider_auto = True
        if self.llm_model is None and self.llm_provider in DEFAULT_LLM_MODELS:
            self.llm_model = DEFAULT_LLM_MODELS[self.llm_provider]
        if self.embedding_model is None:
            self.embedding_model = DEFAULT_EMBEDDING_MODELS[self.embedding_provider]
        if self.min_relevance is None:
            key = (self.embedding_provider, self.embedding_model)
            self.min_relevance = CALIBRATED_MIN_RELEVANCE.get(key, 0.0)
            self._gate_uncalibrated = key not in CALIBRATED_MIN_RELEVANCE
        return self

    @property
    def gate_uncalibrated(self) -> bool:
        """True when the gate was switched off automatically because no calibration
        exists for the embedding model (an explicit RAG_MIN_RELEVANCE=0 is a choice)."""
        return self._gate_uncalibrated

    @property
    def llm_provider_auto(self) -> bool:
        """True when ``llm_provider`` was chosen from the configured keys (auto mode)."""
        return self._llm_provider_auto

    def api_key_for(self, provider: str) -> SecretStr | None:
        return {"anthropic": self.anthropic_api_key, "openai": self.openai_api_key}.get(provider)

    @property
    def collections_dir(self) -> Path:
        return self.data_dir / "collections"


def detect_provider(api_key: str) -> LLMProviderName | None:
    """Guess the vendor from a key's format: Anthropic keys start with ``sk-ant-``;
    OpenAI keys start with ``sk-`` (``sk-proj-``, ``sk-svcacct-``, legacy ``sk-``)."""
    key = api_key.strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-"):
        return "openai"
    return None


def _provider_from_keys(anthropic: SecretStr | None, openai: SecretStr | None) -> str:
    if anthropic is not None and anthropic.get_secret_value().strip():
        return "anthropic"
    if openai is not None and openai.get_secret_value().strip():
        return "openai"
    return "none"
