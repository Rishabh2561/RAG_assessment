"""Per-collection registry of ingested documents."""

from __future__ import annotations

import json
import os
from pathlib import Path

from rag_generator.models import DocumentRecord


class DocumentCatalog:
    FILENAME = "documents.json"

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._records: dict[str, DocumentRecord] = {}
        path = directory / self.FILENAME
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._records = {r["doc_id"]: DocumentRecord(**r) for r in raw}

    def __len__(self) -> int:
        return len(self._records)

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self._records.get(doc_id)

    def find_by_source(self, source: str) -> DocumentRecord | None:
        return next((r for r in self._records.values() if r.source == source), None)

    def list(self) -> list[DocumentRecord]:
        return sorted(self._records.values(), key=lambda r: r.ingested_at)

    def doc_ids(self) -> set[str]:
        return set(self._records)

    def upsert(self, record: DocumentRecord) -> None:
        self._records[record.doc_id] = record

    def remove(self, doc_id: str) -> DocumentRecord | None:
        return self._records.pop(doc_id, None)

    def persist(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / self.FILENAME
        tmp = target.with_suffix(".tmp")
        payload = [r.model_dump(mode="json") for r in self.list()]
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, target)
