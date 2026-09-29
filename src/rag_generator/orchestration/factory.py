"""Composition root: the only place that maps configuration to concrete classes.

Everything else depends on the protocols (``EmbeddingProvider``, ``VectorStore``,
``Retriever``, ``Reranker``, ``LLMProvider``), so swapping an implementation is a
change here plus a new class, never a change to the pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from pydantic import SecretStr

from rag_generator.chunking import RecursiveChunker
from rag_generator.config import RetrievalMode, Settings
from rag_generator.embeddings import EmbeddingProvider, FastEmbedProvider, HashingEmbeddingProvider
from rag_generator.errors import RAGError
from rag_generator.generation import LLMProvider
from rag_generator.ingestion import ParserRegistry
from rag_generator.observability import get_logger, log_event
from rag_generator.orchestration.ingestion_service import IngestionService
from rag_generator.orchestration.query_service import QueryOptions, QueryService
from rag_generator.reranking import CrossEncoderReranker, NoOpReranker, Reranker
from rag_generator.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever
from rag_generator.storage import Collection, CollectionRepository, NumpyVectorStore

logger = get_logger(__name__)

_OPENAI_MISSING = (
    "the OpenAI provider needs the 'openai' package: pip install -e \".[openai]\" "
    "(or uv sync --extra openai)"
)


def _secret(value: SecretStr | None) -> str | None:
    return (value.get_secret_value() or None) if value else None


def build_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(settings.hashing_dimensions)
    if settings.embedding_provider == "openai":
        try:
            import openai  # noqa: F401  (fail fast with a helpful message)

            from rag_generator.embeddings.openai_provider import OpenAIEmbeddingProvider
        except ImportError as exc:
            raise RAGError(_OPENAI_MISSING) from exc
        return OpenAIEmbeddingProvider(
            settings.embedding_model,
            api_key=_secret(settings.openai_api_key),
            base_url=settings.openai_base_url,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    return FastEmbedProvider(
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        cache_dir=settings.model_cache_dir,
    )


def build_repository(settings: Settings, embedder: EmbeddingProvider) -> CollectionRepository:
    model_id = embedder.model_id

    def store_factory(directory: Path) -> NumpyVectorStore:
        return NumpyVectorStore(directory, expected_model_id=model_id)

    return CollectionRepository(settings.collections_dir, store_factory)


def build_retriever(
    settings: Settings,
    collection: Collection,
    embedder: EmbeddingProvider,
    mode: RetrievalMode | None = None,
) -> Retriever:
    mode = mode or settings.retrieval_mode
    if mode == "dense":
        return DenseRetriever(collection, embedder)
    if mode == "bm25":
        return BM25Retriever(collection)
    return HybridRetriever(
        DenseRetriever(collection, embedder),
        BM25Retriever(collection),
        rrf_k=settings.rrf_k,
        pool=settings.candidate_pool,
    )


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "cross_encoder":
        return CrossEncoderReranker(settings.reranker_model, cache_dir=settings.model_cache_dir)
    return NoOpReranker()


def build_llm(settings: Settings) -> LLMProvider | None:
    if settings.llm_provider == "none":
        return None
    if settings.llm_provider == "openai":
        try:
            from rag_generator.generation.openai_provider import OpenAIProvider
        except ImportError as exc:
            raise RAGError(_OPENAI_MISSING) from exc
        return OpenAIProvider(
            settings.llm_model,
            api_key=_secret(settings.openai_api_key),
            base_url=settings.openai_base_url,
            max_tokens=settings.llm_max_tokens,
            reasoning_effort=settings.openai_reasoning_effort,
            temperature=settings.llm_temperature,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    from rag_generator.generation.anthropic_provider import AnthropicProvider

    return AnthropicProvider(
        settings.llm_model,
        api_key=_secret(settings.anthropic_api_key),
        max_tokens=settings.llm_max_tokens,
        effort=settings.llm_effort,
        temperature=settings.llm_temperature,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        server_fallback=settings.anthropic_server_fallback,
    )


def query_options(settings: Settings) -> QueryOptions:
    return QueryOptions(
        top_k=settings.top_k,
        candidate_pool=settings.candidate_pool,
        min_relevance=settings.min_relevance,
        max_context_chars=settings.max_context_chars,
        max_question_chars=settings.max_question_chars,
        query_rewrite=settings.query_rewrite,
        max_rewrites=settings.max_rewrites,
        rrf_k=settings.rrf_k,
    )


class RAGApplication:
    """Long-lived container of shared components (models are loaded once)."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: EmbeddingProvider | None = None,
        llm: LLMProvider | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        if settings.gate_uncalibrated:
            log_event(
                logger,
                "relevance_gate_disabled",
                level=logging.WARNING,
                embedding_model=settings.embedding_model,
                hint="no calibrated RAG_MIN_RELEVANCE for this model; run `rag eval` and set it",
            )
        self.embedder = embedder or build_embedder(settings)
        self.llm = llm if llm is not None else build_llm(settings)
        self.reranker = reranker or build_reranker(settings)
        self.repository = build_repository(settings, self.embedder)
        self.registry = ParserRegistry()
        self.chunker = RecursiveChunker(
            settings.chunk_size, settings.chunk_overlap, settings.min_chunk_chars
        )

    def ingestion(self) -> IngestionService:
        return IngestionService(
            self.repository,
            self.registry,
            self.chunker,
            self.embedder,
            max_file_bytes=int(self.settings.max_file_mb * 1024 * 1024),
        )

    def query(
        self,
        collection_name: str,
        *,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
        use_llm: bool = True,
    ) -> QueryService:
        collection = self.repository.open(collection_name)
        options = query_options(self.settings)
        if top_k is not None:
            options = replace(options, top_k=top_k)
        return QueryService(
            collection,
            build_retriever(self.settings, collection, self.embedder, mode),
            self.reranker,
            self.llm if use_llm else None,
            options,
        )
