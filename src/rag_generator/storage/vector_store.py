"""Vector store interface and the in-process NumPy implementation (ADR-004)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from rag_generator.errors import IndexMismatchError, RAGError
from rag_generator.models import Chunk

SCHEMA_VERSION = 1


class VectorStore(Protocol):
    """Stores chunks with their vectors for one collection."""

    @property
    def version(self) -> int:
        """Monotonic counter bumped on every mutation (for derived-index caching)."""
        ...

    @property
    def model_id(self) -> str | None:
        """Embedding model the stored vectors came from (None if empty)."""
        ...

    def __len__(self) -> int: ...

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def delete_document(self, doc_id: str) -> int: ...

    def drop_orphans(self, valid_doc_ids: set[str]) -> int: ...

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def persist(self) -> None: ...


class NumpyVectorStore:
    """Exact cosine search over an in-memory float32 matrix.

    Chunks and vectors are persisted together in a single ``index.npz`` so they can
    never be written out of step. State is swapped as one tuple so concurrent readers
    always see a consistent (chunks, vectors) pair.
    """

    FILENAME = "index.npz"

    def __init__(self, directory: Path, expected_model_id: str) -> None:
        self.directory = directory
        self.expected_model_id = expected_model_id
        self._state: tuple[list[Chunk], np.ndarray | None] = ([], None)
        self._model_id: str | None = None
        self._version = 0
        self._load()

    # --- Properties -----------------------------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def __len__(self) -> int:
        return len(self._state[0])

    # --- Mutations ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(f"expected ({len(chunks)}, dim) vectors, got {vectors.shape}")
        self._ensure_compatible()
        current_chunks, current_vectors = self._state
        if current_vectors is not None and current_vectors.shape[1] != vectors.shape[1]:
            raise IndexMismatchError(
                f"vector dimension {vectors.shape[1]} != index dimension {current_vectors.shape[1]}"
            )
        new_vectors = vectors if current_vectors is None else np.vstack([current_vectors, vectors])
        self._state = (current_chunks + list(chunks), new_vectors)
        self._model_id = self.expected_model_id
        self._version += 1

    def delete_document(self, doc_id: str) -> int:
        chunks, vectors = self._state
        keep = [i for i, c in enumerate(chunks) if c.doc_id != doc_id]
        removed = len(chunks) - len(keep)
        if removed:
            self._state = (
                [chunks[i] for i in keep],
                vectors[keep] if vectors is not None and keep else None,
            )
            if not keep:
                self._model_id = None
            self._version += 1
        return removed

    def drop_orphans(self, valid_doc_ids: set[str]) -> int:
        """Remove chunks whose document is not in the catalog (crash recovery)."""
        orphan_ids = {c.doc_id for c in self._state[0]} - valid_doc_ids
        return sum(self.delete_document(doc_id) for doc_id in orphan_ids)

    # --- Queries --------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        chunks, vectors = self._state
        if vectors is None or not chunks:
            return []
        self._ensure_compatible()
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != vectors.shape[1]:
            raise IndexMismatchError(
                f"query dimension {query.shape[0]} != index dimension {vectors.shape[1]}"
            )
        scores = vectors @ query
        k = min(k, len(chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(chunks[i], float(scores[i])) for i in top]

    def all_chunks(self) -> list[Chunk]:
        return list(self._state[0])

    # --- Persistence ----------------------------------------------------------------

    def persist(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        chunks, vectors = self._state
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model_id": self._model_id,
            "num_chunks": len(chunks),
            "dimensions": int(vectors.shape[1]) if vectors is not None else None,
        }
        chunks_json = json.dumps([c.model_dump() for c in chunks])
        target = self.directory / self.FILENAME
        tmp = target.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            vectors=vectors if vectors is not None else np.zeros((0, 0), dtype=np.float32),
            chunks=np.array(chunks_json),
            manifest=np.array(json.dumps(manifest)),
        )
        os.replace(tmp, target)

    def _load(self) -> None:
        path = self.directory / self.FILENAME
        if not path.exists():
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                manifest = json.loads(str(data["manifest"]))
                chunks = [Chunk(**c) for c in json.loads(str(data["chunks"]))]
                vectors = data["vectors"].astype(np.float32)
        except Exception as exc:
            raise RAGError(
                f"index at {path} is unreadable ({exc}); re-ingest the collection"
            ) from exc
        if manifest.get("schema_version") != SCHEMA_VERSION or len(chunks) != len(vectors):
            raise RAGError(f"index at {path} is inconsistent; re-ingest the collection")
        self._state = (chunks, vectors if chunks else None)
        self._model_id = manifest.get("model_id")

    def _ensure_compatible(self) -> None:
        if self._model_id is not None and self._model_id != self.expected_model_id:
            raise IndexMismatchError(
                f"collection was built with embedding model '{self._model_id}' but the "
                f"configured model is '{self.expected_model_id}'. Re-ingest the collection "
                "or restore the original RAG_EMBEDDING_* settings."
            )
