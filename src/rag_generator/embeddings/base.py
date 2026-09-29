"""Embedding interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    """Maps text to L2-normalised float32 vectors.

    ``model_id`` identifies the vector space; the vector store records it so that a
    collection is never queried with vectors from a different model.
    """

    @property
    def model_id(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return an array of shape ``(dim,)``."""
        ...


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
