"""Ingestion pipeline: bytes → parse → chunk → embed → store (write path only)."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

from rag_generator.chunking import RecursiveChunker
from rag_generator.embeddings import EmbeddingProvider
from rag_generator.errors import (
    DuplicateSourceError,
    EmptyDocumentError,
    FileTooLargeError,
    RAGError,
)
from rag_generator.ingestion import ParserRegistry
from rag_generator.models import DocumentRecord, IngestReport, IngestResult
from rag_generator.observability import StageTimer, get_logger, log_event
from rag_generator.storage import Collection, CollectionRepository

logger = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        repository: CollectionRepository,
        registry: ParserRegistry,
        chunker: RecursiveChunker,
        embedder: EmbeddingProvider,
        max_file_bytes: int,
    ) -> None:
        self.repository = repository
        self.registry = registry
        self.chunker = chunker
        self.embedder = embedder
        self.max_file_bytes = max_file_bytes

    # --- Public API -----------------------------------------------------------------

    def ingest_paths(self, paths: Iterable[Path], collection_name: str) -> IngestReport:
        """Ingest files and directories. Directories are walked recursively and only
        supported file types are picked up; explicitly named files are always tried.

        Source names are paths relative to the common parent of the inputs, so
        ``a/notes.md`` and ``b/notes.md`` stay distinct while a single directory's files
        keep their plain relative names."""
        paths = list(paths)
        files: list[Path] = []
        for path in paths:
            files.extend(self._walk(path) if path.is_dir() else [path])
        roots = [str((p if p.is_dir() else p.parent).resolve()) for p in paths]
        base = Path(os.path.commonpath(roots)) if roots else None
        items = [(_relative_source(f, base), f) for f in files]
        return self._ingest(items, collection_name)

    def ingest_bytes(self, files: list[tuple[str, bytes]], collection_name: str) -> IngestReport:
        """Ingest in-memory files, e.g. HTTP uploads, as ``(source_name, data)`` pairs."""
        return self._ingest(iter(files), collection_name)

    def delete_document(self, collection_name: str, doc_id: str) -> DocumentRecord | None:
        collection = self.repository.open(collection_name)
        with collection.lock:
            record = collection.catalog.remove(doc_id)
            if record is None:
                return None
            collection.store.delete_document(doc_id)
            collection.persist()
        log_event(logger, "document_deleted", collection=collection_name, doc_id=doc_id)
        return record

    # --- Internals ------------------------------------------------------------------

    def _walk(self, directory: Path) -> list[Path]:
        found, skipped = [], 0
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if self.registry.is_supported(path.name):
                found.append(path)
            else:
                skipped += 1
        if skipped:
            log_event(logger, "unsupported_files_skipped", directory=str(directory), count=skipped)
        return found

    def _ingest(
        self, items: Iterable[tuple[str, bytes | Path]], collection_name: str
    ) -> IngestReport:
        start = time.perf_counter()
        collection = self.repository.open(collection_name, create=True)
        results: list[IngestResult] = []
        seen_sources: set[str] = set()
        with collection.lock:
            version_before = collection.store.version
            for source, payload in items:
                if source in seen_sources:
                    error = DuplicateSourceError(
                        f"{source}: another file in this batch has the same name; "
                        "rename one or ingest them separately"
                    )
                    results.append(_failed(source, error))
                    continue
                seen_sources.add(source)
                results.append(self._ingest_one(collection, source, payload))
            if collection.store.version != version_before:
                collection.persist()
        report = IngestReport(
            collection=collection_name,
            results=results,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        log_event(
            logger,
            "ingest_batch_completed",
            collection=collection_name,
            files=len(results),
            succeeded=report.succeeded,
            failed=report.failed,
            total_chunks=len(collection.store),
            duration_ms=round(report.duration_ms, 1),
        )
        return report

    def _ingest_one(
        self, collection: Collection, source: str, payload: bytes | Path
    ) -> IngestResult:
        try:
            data = payload.read_bytes() if isinstance(payload, Path) else payload
            if len(data) > self.max_file_bytes:
                raise FileTooLargeError(
                    f"{source}: {len(data) / 1e6:.1f} MB exceeds the "
                    f"{self.max_file_bytes / 1e6:.0f} MB limit"
                )
            content_hash = hashlib.sha256(data).hexdigest()
            existing = collection.catalog.get(content_hash[:16])
            if existing is not None:
                return self._skip_duplicate(collection, source, existing)
            return self._index(collection, source, data, content_hash)
        except (RAGError, OSError) as exc:
            return _failed(source, exc)
        except Exception as exc:  # parser/library bugs must not abort the whole batch
            logger.exception("unexpected ingestion error", extra={"fields": {"source": source}})
            return _failed(source, exc)

    def _skip_duplicate(
        self, collection: Collection, source: str, existing: DocumentRecord
    ) -> IngestResult:
        # Identical content is already indexed. If this name previously held *other*
        # content, that stale version must go, or it would stay citable under this name.
        stale = collection.catalog.find_by_source(source)
        if stale is not None and stale.doc_id != existing.doc_id:
            collection.store.delete_document(stale.doc_id)
            collection.catalog.remove(stale.doc_id)
        log_event(logger, "document_skipped_duplicate", source=source, doc_id=existing.doc_id)
        message = None if existing.source == source else f"same content as {existing.source}"
        return IngestResult(
            source=source, status="skipped_duplicate", doc_id=existing.doc_id, message=message
        )

    def _index(
        self, collection: Collection, source: str, data: bytes, content_hash: str
    ) -> IngestResult:
        timer = StageTimer()
        doc_id = content_hash[:16]
        with timer.stage("parse"):
            parsed = self.registry.parse(data, source)
        with timer.stage("chunk"):
            chunks = self.chunker.chunk(parsed, doc_id)
        if not chunks:
            raise EmptyDocumentError(f"{source}: no chunks produced (text too short)")
        with timer.stage("embed"):
            vectors = self.embedder.embed_documents([c.text for c in chunks])

        # Add the new version *before* removing the old one: if add() fails (e.g. an
        # embedding-model mismatch) the previous version stays intact.
        collection.store.delete_document(doc_id)  # clears orphans left by a crash
        collection.store.add(chunks, vectors)
        previous = collection.catalog.find_by_source(source)
        if previous is not None:
            collection.store.delete_document(previous.doc_id)
            collection.catalog.remove(previous.doc_id)
        pages = {s.page for s in parsed.sections if s.page is not None}
        collection.catalog.upsert(
            DocumentRecord(
                doc_id=doc_id,
                source=source,
                file_type=parsed.file_type,
                content_hash=content_hash,
                size_bytes=len(data),
                num_chunks=len(chunks),
                num_pages=max(pages) if pages else None,
            )
        )
        status = "replaced" if previous is not None else "ingested"
        log_event(
            logger,
            "document_ingested",
            collection=collection.name,
            source=source,
            doc_id=doc_id,
            status=status,
            chunks=len(chunks),
            chars=parsed.char_count,
            **{f"{name}_ms": round(ms, 1) for name, ms in timer.timings},
        )
        return IngestResult(
            source=source,
            status=status,
            doc_id=doc_id,
            num_chunks=len(chunks),
            warnings=parsed.warnings,
        )


def _failed(source: str, exc: Exception) -> IngestResult:
    log_event(
        logger,
        "document_failed",
        level=logging.WARNING,
        source=source,
        error_type=type(exc).__name__,
        error=str(exc),
    )
    return IngestResult(
        source=source, status="failed", error_type=type(exc).__name__, message=str(exc)
    )


def _relative_source(path: Path, base: Path | None) -> str:
    if base is None:
        return path.name
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.name
