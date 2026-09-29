"""Reranker interface and the default no-op implementation (ADR-006)."""

from __future__ import annotations

from typing import Protocol

from rag_generator.models import RetrievedChunk


class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]: ...


class NoOpReranker:
    name = "none"

    def rerank(self, query: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        return candidates[:k]
