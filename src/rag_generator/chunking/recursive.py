"""Recursive, page-bounded chunking (ADR-002).

Text is split on the coarsest boundary that exists (blank line → newline → sentence end
→ space), pieces that are still too large are split on the next boundary, and the pieces
are then greedily merged back up to ``chunk_size`` with ``chunk_overlap`` characters of
trailing context carried into the next chunk. Chunks never cross a section (page).
"""

from __future__ import annotations

import re

from rag_generator.models import Chunk, ParsedDocument

# Ordered coarse → fine. Each pattern matches the boundary *after* which we may split.
_SEPARATORS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n\n"),
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?])\s+"),
    re.compile(r" "),
)


class RecursiveChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150, min_chars: int = 20):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chars = min_chars

    def chunk(self, document: ParsedDocument, doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in document.sections:
            for text in self.split_text(section.text):
                if len(text) < self.min_chars:
                    continue
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc_id, index),
                        doc_id=doc_id,
                        source=document.source,
                        page=section.page,
                        index=index,
                        text=text,
                    )
                )
        return chunks

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        pieces = self._split_recursive(text, 0)
        return self._absorb_small(self._merge(pieces))

    def _absorb_small(self, chunks: list[str]) -> list[str]:
        """Append chunks shorter than ``min_chars`` to their predecessor.

        A short trailing line ("Contact: x@y.com") can end up alone when the previous
        chunk leaves no overlap; dropping it would lose text, so it is kept with the
        preceding chunk (a small overflow of ``chunk_size`` is accepted). Only a section
        whose *entire* text is shorter than ``min_chars`` is dropped, in ``chunk``.
        """
        result: list[str] = []
        for chunk in chunks:
            if result and len(chunk) < self.min_chars:
                result[-1] = f"{result[-1]}\n{chunk}"
            else:
                result.append(chunk)
        return result

    def _split_recursive(self, text: str, level: int) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if level >= len(_SEPARATORS):
            step = self.chunk_size - self.chunk_overlap
            return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]
        pieces: list[str] = []
        for piece in _split_keep_separator(text, _SEPARATORS[level]):
            if len(piece) > self.chunk_size:
                pieces.extend(self._split_recursive(piece, level + 1))
            else:
                pieces.append(piece)
        return pieces

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for piece in pieces:
            if current and current_len + len(piece) > self.chunk_size:
                chunks.append("".join(current).strip())
                current, current_len = self._overlap_tail(current)
            current.append(piece)
            current_len += len(piece)
        if current:
            chunks.append("".join(current).strip())
        return [c for c in chunks if c]

    def _overlap_tail(self, pieces: list[str]) -> tuple[list[str], int]:
        """Trailing pieces of the previous chunk whose total length fits in the overlap."""
        tail: list[str] = []
        length = 0
        for piece in reversed(pieces):
            if length + len(piece) > self.chunk_overlap:
                break
            tail.insert(0, piece)
            length += len(piece)
        return tail, length


def _split_keep_separator(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split on ``pattern`` keeping each separator attached to the preceding piece."""
    pieces: list[str] = []
    start = 0
    for match in pattern.finditer(text):
        end = match.end()
        if end > start:
            pieces.append(text[start:end])
        start = end
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def make_chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id[:12]}-{index:05d}"
