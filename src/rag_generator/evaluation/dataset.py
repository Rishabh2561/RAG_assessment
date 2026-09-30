"""Evaluation dataset schema and loader.

One JSON object per line::

    {"id": "q01", "question": "...", "answerable": true,
     "expected_sources": ["handbook.md"],
     "expected_evidence": ["25 days of annual leave"],
     "expected_keywords": ["25"], "category": "factual"}

A retrieved chunk counts as *relevant* when its source is in ``expected_sources`` and,
if ``expected_evidence`` is given, its text contains at least one evidence string
(case-insensitive). Labelling by evidence text rather than chunk id keeps labels valid
when chunking parameters change.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from rag_generator.models import Chunk


class EvalItem(BaseModel):
    id: str
    question: str
    answerable: bool = True
    expected_sources: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    category: str = "general"

    def is_relevant(self, chunk: Chunk) -> bool:
        if chunk.source not in self.expected_sources:
            return False
        if not self.expected_evidence:
            return True
        text = " ".join(chunk.text.lower().split())
        return any(" ".join(e.lower().split()) in text for e in self.expected_evidence)


def load_dataset(path: Path) -> list[EvalItem]:
    return parse_dataset(path.read_text(encoding="utf-8"), origin=str(path))


def parse_dataset(text: str, origin: str = "dataset") -> list[EvalItem]:
    """Parse JSONL eval items; ``origin`` names the source in error messages."""
    items = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            items.append(EvalItem.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{origin}:{line_no}: invalid eval item ({exc})") from exc
    if not items:
        raise ValueError(f"{origin}: no evaluation items found")
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{origin}: duplicate item ids")
    return items
