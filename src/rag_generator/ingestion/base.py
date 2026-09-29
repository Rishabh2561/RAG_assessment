"""Parser interface."""

from __future__ import annotations

from typing import Protocol

from rag_generator.models import ParsedDocument


class DocumentParser(Protocol):
    """Turns raw file bytes into a :class:`ParsedDocument`.

    Implementations raise subclasses of :class:`~rag_generator.errors.IngestionError`
    for unusable input and must never return a document with no text.
    """

    extensions: tuple[str, ...]

    def parse(self, data: bytes, source: str) -> ParsedDocument: ...
