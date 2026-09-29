"""Citation validation and resolution (ADR-010)."""

from __future__ import annotations

import re

from rag_generator.models import Citation, RetrievedChunk

_INLINE_MARKER = re.compile(r"\[(S\d+)\]")
SNIPPET_CHARS = 240


def extract_inline_ids(answer: str) -> list[str]:
    """Source ids cited inline as ``[S#]``, in first-appearance order."""
    return list(dict.fromkeys(_INLINE_MARKER.findall(answer)))


def resolve_citations(
    declared_ids: list[str], answer: str, source_map: dict[str, RetrievedChunk]
) -> tuple[list[Citation], list[str]]:
    """Validate cited ids against the sources actually supplied to the model.

    The union of the structured ``citations`` list and inline ``[S#]`` markers is
    considered, so a model that cites inline but forgets the list is not penalised.
    Returns (valid citations in citation order, invalid ids).
    """
    candidate_ids = list(dict.fromkeys([*declared_ids, *extract_inline_ids(answer)]))
    citations: list[Citation] = []
    invalid: list[str] = []
    for sid in candidate_ids:
        sid = sid.strip().strip("[]")
        passage = source_map.get(sid)
        if passage is None:
            invalid.append(sid)
            continue
        chunk = passage.chunk
        citations.append(
            Citation(
                source_id=sid,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.source,
                page=chunk.page,
                snippet=_snippet(chunk.text),
            )
        )
    return citations, invalid


def strip_invalid_markers(answer: str, invalid_ids: list[str]) -> str:
    """Remove inline markers pointing at sources that were never supplied."""
    for sid in invalid_ids:
        answer = answer.replace(f"[{sid}]", "")
    return answer


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 3].rstrip() + "..."
