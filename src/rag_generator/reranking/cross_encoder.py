"""Local cross-encoder reranker via fastembed (off by default; ADR-006)."""

from __future__ import annotations

from pathlib import Path

from rag_generator.errors import RAGError
from rag_generator.models import RetrievedChunk


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def rerank(self, query: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._load()
        scores = list(model.rerank(query, [c.chunk.text for c in candidates]))
        rescored = [
            c.model_copy(update={"rerank_score": float(s), "score": float(s)})
            for c, s in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda c: -c.score)
        return rescored[:k]

    def _load(self):
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(
                    model_name=self.model_name,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                )
            except Exception as exc:
                raise RAGError(f"could not load reranker '{self.model_name}': {exc}") from exc
        return self._model
