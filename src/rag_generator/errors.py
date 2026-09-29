"""Error hierarchy.

Every error the application raises deliberately derives from ``RAGError`` so that the
interfaces can map them to user-facing messages / HTTP statuses in one place.
"""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all expected application errors."""


# --- Ingestion ---------------------------------------------------------------------


class IngestionError(RAGError):
    """A single document could not be ingested."""


class UnsupportedFileTypeError(IngestionError):
    pass


class EmptyDocumentError(IngestionError):
    pass


class CorruptDocumentError(IngestionError):
    pass


class OCRRequiredError(IngestionError):
    """The document appears to be scanned images with no extractable text."""


class FileTooLargeError(IngestionError):
    pass


class DuplicateSourceError(IngestionError):
    """Two files in one batch map to the same source name."""


# --- Components --------------------------------------------------------------------


class EmbeddingError(RAGError):
    pass


class IndexMismatchError(RAGError):
    """The collection was built with a different embedding model or dimension."""


class GenerationError(RAGError):
    """The LLM call failed or returned an unusable response."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


# --- Collections / queries -----------------------------------------------------------


class CollectionNotFoundError(RAGError):
    pass


class CollectionEmptyError(RAGError):
    pass


class DocumentNotFoundError(RAGError):
    pass


class InvalidQueryError(RAGError):
    pass
