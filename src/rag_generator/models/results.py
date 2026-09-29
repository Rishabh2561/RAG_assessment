"""Query-side data models (returned by the CLI and API)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rag_generator.models.documents import Chunk

GroundingStatus = Literal["grounded", "unverified", "abstained", "retrieval_only"]


class RetrievedChunk(BaseModel):
    """A chunk with the scores that led to its selection."""

    chunk: Chunk
    score: float  # score used for the final ranking (mode-dependent)
    dense_score: float | None = None  # cosine similarity, when available
    lexical_score: float | None = None  # BM25 score, when available
    rerank_score: float | None = None


class Citation(BaseModel):
    source_id: str  # "S1" etc. as shown to the LLM
    chunk_id: str
    doc_id: str
    source: str
    page: int | None
    snippet: str


class StageTiming(BaseModel):
    stage: str
    duration_ms: float


class QueryTrace(BaseModel):
    """Everything needed to explain how an answer was produced (no document text)."""

    collection: str
    retrieval_mode: str
    reranker: str
    llm_model: str | None
    attempts: int = 0
    rewritten_queries: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    best_dense_score: float | None = None
    gate_threshold: float | None = None
    invalid_citations: list[str] = Field(default_factory=list)
    rewrite_error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    timings: list[StageTiming] = Field(default_factory=list)
    total_ms: float = 0.0


class Answer(BaseModel):
    question: str
    answer: str | None
    answerable: bool
    grounding_status: GroundingStatus
    citations: list[Citation] = Field(default_factory=list)
    reason: str | None = None  # why the system abstained, when it did
    passages: list[RetrievedChunk] = Field(default_factory=list)
    trace: QueryTrace
