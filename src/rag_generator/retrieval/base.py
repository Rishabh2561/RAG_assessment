"""Retriever interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_generator.models import RetrievedChunk


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    # Highest dense cosine seen among candidates; drives the relevance gate.
    # None when the mode has no dense component (bm25).
    best_dense_score: float | None


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> RetrievalResult: ...
