"""Okapi BM25 over chunk texts, with an inverted index for sparse scoring."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from rag_generator.models import Chunk
from rag_generator.textproc import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_lengths: list[int] = []
        for i, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            self._doc_lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings[term].append((i, tf))
        n = len(chunks)
        self._avg_len = (sum(self._doc_lengths) / n) if n else 0.0
        # BM25+ style non-negative IDF (Lucene variant).
        self._idf = {
            term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self._postings.items()
        }

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in self._postings[term]:
                norm = 1 - self.b + self.b * self._doc_lengths[i] / (self._avg_len or 1.0)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self.chunks[i], score) for i, score in ranked]
