"""Ingestion pipeline: bytes → parse → chunk → embed → store (write path only)."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterable
from pathlib import Path

from rag_generator.chunking import RecursiveChunker
from rag_generator.embeddings import EmbeddingProvider
from rag_generator.errors import EmptyDocumentError, FileTooLargeError, RAGError
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
        supported file types are picked up; explicitly named files are always tried."""
        items: list[tuple[str, Path]] = []
        for path in paths:
            if path.is_dir():
                items.extend(self._walk(path))
            else:
                items.append((path.name, path))
        return self._ingest(((source, path) for source, path in items), collection_name)

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

    def _walk(self, directory: Path) -> list[tuple[str, Path]]:
        found, skipped = [], 0
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if self.registry.is_supported(path.name):
                found.append((path.relative_to(directory).as_posix(), path))
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
        with collection.lock:
            for source, payload in items:
                results.append(self._ingest_one(collection, source, payload))
            if any(r.status in ("ingested", "replaced") for r in results):
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
        timer = StageTimer()
        try:
            data = payload.read_bytes() if isinstance(payload, Path) else payload
            if len(data) > self.max_file_bytes:
                raise FileTooLargeError(
                    f"{source}: {len(data) / 1e6:.1f} MB exceeds the "
                    f"{self.max_file_bytes / 1e6:.0f} MB limit"
                )
            content_hash = hashlib.sha256(data).hexdigest()
            doc_id = content_hash[:16]
            if collection.catalog.get(doc_id) is not None:
                log_event(logger, "document_skipped_duplicate", source=source, doc_id=doc_id)
                return IngestResult(source=source, status="skipped_duplicate", doc_id=doc_id)

            with timer.stage("parse"):
                parsed = self.registry.parse(data, source)
            with timer.stage("chunk"):
                chunks = self.chunker.chunk(parsed, doc_id)
            if not chunks:
                raise EmptyDocumentError(f"{source}: no chunks produced (text too short)")
            with timer.stage("embed"):
                vectors = self.embedder.embed_documents([c.text for c in chunks])

            previous = collection.catalog.find_by_source(source)
            if previous is not None:
                collection.store.delete_document(previous.doc_id)
                collection.catalog.remove(previous.doc_id)
            collection.store.delete_document(doc_id)  # idempotent re-add after a crash
            collection.store.add(chunks, vectors)
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
        except (RAGError, OSError) as exc:
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
