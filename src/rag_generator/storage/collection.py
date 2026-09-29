"""Collections: an isolated (vector store + catalog) pair per document set."""

from __future__ import annotations

import re
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_generator.errors import CollectionNotFoundError, InvalidQueryError
from rag_generator.observability import get_logger, log_event
from rag_generator.storage.catalog import DocumentCatalog
from rag_generator.storage.vector_store import VectorStore

logger = get_logger(__name__)

_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
StoreFactory = Callable[[Path], VectorStore]


class Collection:
    """Handle to one collection. Writers must hold ``lock``; readers need not."""

    def __init__(self, name: str, directory: Path, store: VectorStore, catalog: DocumentCatalog):
        self.name = name
        self.directory = directory
        self.store = store
        self.catalog = catalog
        self.lock = threading.RLock()
        self.closed = False
        self._derived: dict[str, tuple[int, Any]] = {}

    def derived(self, key: str, build: Callable[[], Any]) -> Any:
        """Cache state derived from the store (e.g. a BM25 index) until the store changes."""
        cached = self._derived.get(key)
        if cached is not None and cached[0] == self.store.version:
            return cached[1]
        # Capture the version *before* building: if a writer mutates the store while we
        # build, the cached value is tagged with the older version and rebuilt next time.
        version = self.store.version
        value = build()
        self._derived[key] = (version, value)
        return value

    def persist(self) -> None:
        # Index first, catalog second. A crash in between leaves the two out of step;
        # CollectionRepository.open reconciles both directions on the next start.
        if self.closed:
            raise CollectionNotFoundError(f"collection '{self.name}' was dropped")
        self.store.persist()
        self.catalog.persist()


class CollectionRepository:
    """Opens, caches and deletes collections under a root directory."""

    def __init__(self, root: Path, store_factory: StoreFactory) -> None:
        self.root = root
        self.store_factory = store_factory
        self._open: dict[str, Collection] = {}
        self._lock = threading.Lock()

    def open(self, name: str, *, create: bool = False) -> Collection:
        validate_collection_name(name)
        with self._lock:
            if name in self._open:
                return self._open[name]
            directory = self.root / name
            if not directory.exists() and not create:
                raise CollectionNotFoundError(
                    f"collection '{name}' does not exist; ingest documents into it first"
                )
            catalog = DocumentCatalog(directory)
            store = self.store_factory(directory)
            _reconcile(name, store, catalog)
            collection = Collection(name, directory, store, catalog)
            self._open[name] = collection
            return collection

    def list_names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def drop(self, name: str) -> None:
        validate_collection_name(name)
        directory = self.root / name
        if not directory.exists():
            raise CollectionNotFoundError(f"collection '{name}' does not exist")
        with self._lock:
            collection = self._open.pop(name, None)
            if collection is None:
                shutil.rmtree(directory)
            else:
                with collection.lock:  # wait for in-flight writes, then refuse new ones
                    collection.closed = True
                    shutil.rmtree(directory)
        log_event(logger, "collection_dropped", collection=name)


def _reconcile(name: str, store: VectorStore, catalog: DocumentCatalog) -> None:
    """Repair a collection left inconsistent by a crash between the two writes.

    Chunks without a catalog entry (crash during add) are dropped; catalog entries
    without chunks (crash during delete/replace) are removed so the file can be
    re-ingested instead of being reported as a duplicate forever.
    """
    if removed := store.drop_orphans(catalog.doc_ids()):
        log_event(logger, "orphan_chunks_dropped", collection=name, chunks=removed)
    indexed = {c.doc_id for c in store.all_chunks()}
    ghosts = [doc_id for doc_id in catalog.doc_ids() if doc_id not in indexed]
    for doc_id in ghosts:
        catalog.remove(doc_id)
    if ghosts:
        log_event(logger, "ghost_catalog_entries_dropped", collection=name, documents=len(ghosts))


def validate_collection_name(name: str) -> None:
    # Also prevents path traversal: names become directory names.
    if not _VALID_NAME.match(name):
        raise InvalidQueryError(
            f"invalid collection name '{name}': use 1-64 letters, digits, '_' or '-'"
        )
