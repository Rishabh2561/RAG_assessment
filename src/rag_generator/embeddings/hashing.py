"""Deterministic, dependency-free embedder for tests and offline smoke runs.

Feature-hashes word unigrams and bigrams into a fixed-size vector. It captures lexical
overlap only (no semantics), so it is *not* a quality embedder; it exists so the whole
pipeline can be exercised without downloading a model.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise

import numpy as np

from rag_generator.embeddings.base import l2_normalise
from rag_generator.textproc import tokenize


class HashingEmbeddingProvider:
    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"hashing:{self.dimensions}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return l2_normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalise(self._vector(text))

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            sign = 1.0 if value & 1 else -1.0
            vector[(value >> 1) % self.dimensions] += sign
        return vector
