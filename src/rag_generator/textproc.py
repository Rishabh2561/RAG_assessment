"""Tokenisation shared by lexical retrieval and the hashing embedder."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+", re.UNICODE)

# Small, conservative English stop-word list: removes function words that dominate
# BM25 term counts without carrying meaning. Kept short so domain terms are never lost.
# fmt: off
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "our", "should", "so", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will",
    "with", "would", "you", "your",
})
# fmt: on


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
