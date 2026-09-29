"""Local ONNX embeddings via fastembed (ADR-003)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rag_generator.embeddings.base import l2_normalise
from rag_generator.errors import EmbeddingError
from rag_generator.observability import get_logger, log_event

logger = get_logger(__name__)


class FastEmbedProvider:
    """Wraps ``fastembed.TextEmbedding``; the model is loaded lazily on first use."""

    def __init__(
        self, model_name: str, batch_size: int = 32, cache_dir: Path | None = None
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None

    @property
    def model_id(self) -> str:
        return f"fastembed:{self.model_name}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("embed_documents called with no texts")
        model = self._load()
        try:
            vectors = list(model.passage_embed(texts, batch_size=self.batch_size))
        except Exception as exc:
            raise EmbeddingError(f"embedding failed: {exc}") from exc
        return l2_normalise(np.stack(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        try:
            vector = next(iter(model.query_embed(text)))
        except Exception as exc:
            raise EmbeddingError(f"query embedding failed: {exc}") from exc
        return l2_normalise(vector)

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                start = time.perf_counter()
                self._model = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                )
                log_event(
                    logger,
                    "embedding_model_loaded",
                    model=self.model_name,
                    load_ms=round((time.perf_counter() - start) * 1000, 1),
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"could not load embedding model '{self.model_name}': {exc}. "
                    "Check the model name and network access for the first download."
                ) from exc
        return self._model
