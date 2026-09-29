"""Hosted OpenAI embeddings (opt-in; ADR-014).

Unlike the local default, this sends document text to OpenAI at ingest time and every
question at query time. Vectors are re-normalised so cosine search works unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rag_generator.embeddings.base import l2_normalise
from rag_generator.errors import EmbeddingError


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 128,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = client
        self._client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout_s,
            "max_retries": max_retries,
        }

    @property
    def model_id(self) -> str:
        return f"openai:{self.model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("embed_documents called with no texts")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed(texts[start : start + self.batch_size]))
        return l2_normalise(np.array(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalise(np.array(self._embed([text])[0], dtype=np.float32))

    def _embed(self, batch: list[str]) -> list[list[float]]:
        import openai

        try:
            if self._client is None:
                self._client = openai.OpenAI(**self._client_kwargs)
            response = self._client.embeddings.create(model=self.model, input=batch)
        except openai.OpenAIError as exc:
            raise EmbeddingError(
                f"OpenAI embedding request failed ({type(exc).__name__}): {exc}. "
                "Check OPENAI_API_KEY and RAG_EMBEDDING_MODEL."
            ) from exc
        # The API returns one item per input with an explicit index; don't assume order.
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(batch):
            raise EmbeddingError(f"expected {len(batch)} embeddings, got {len(items)}")
        return [item.embedding for item in items]
