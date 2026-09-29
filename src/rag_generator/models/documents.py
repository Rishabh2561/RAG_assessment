"""Ingestion-side data models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    """A contiguous span of extracted text (a PDF page, or a whole text file)."""

    model_config = ConfigDict(frozen=True)

    text: str
    page: int | None = None  # 1-based; None for formats without pages


class ParsedDocument(BaseModel):
    """Output of a parser: ordered sections plus minimal metadata."""

    model_config = ConfigDict(frozen=True)

    source: str  # display name, e.g. "handbook.pdf"
    file_type: str  # normalised extension without dot, e.g. "pdf"
    sections: list[Section]
    warnings: list[str] = Field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


class Chunk(BaseModel):
    """A retrievable unit of text with provenance."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    source: str
    page: int | None
    index: int  # position of the chunk within its document
    text: str


class DocumentRecord(BaseModel):
    """Catalog entry for an ingested document."""

    doc_id: str
    source: str
    file_type: str
    content_hash: str
    size_bytes: int
    num_chunks: int
    num_pages: int | None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


IngestStatus = Literal["ingested", "replaced", "skipped_duplicate", "failed"]


class IngestResult(BaseModel):
    """Outcome for one file in an ingestion batch."""

    source: str
    status: IngestStatus
    doc_id: str | None = None
    num_chunks: int = 0
    error_type: str | None = None
    message: str | None = None
    warnings: list[str] = Field(default_factory=list)


class IngestReport(BaseModel):
    collection: str
    results: list[IngestResult]
    duration_ms: float

    @property
    def succeeded(self) -> int:
        return sum(r.status in ("ingested", "replaced") for r in self.results)

    @property
    def failed(self) -> int:
        return sum(r.status == "failed" for r in self.results)
