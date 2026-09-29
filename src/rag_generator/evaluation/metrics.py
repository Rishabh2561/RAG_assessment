"""Pure metric functions (no I/O), so they can be unit-tested against hand-computed values."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    """1.0 if any of the first k results is relevant."""
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def source_recall_at_k(retrieved_sources: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Fraction of expected source documents that appear in the first k results."""
    if not expected:
        return 1.0
    found = set(retrieved_sources[:k])
    return sum(1 for s in expected if s in found) / len(expected)


def keyword_recall(answer: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 1.0
    text = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in text) / len(keywords)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def best_threshold(positives: Sequence[float], negatives: Sequence[float]) -> tuple[float, float]:
    """Threshold on a score that best separates positives (≥) from negatives (<).

    Returns (threshold, accuracy). Candidate thresholds are midpoints between sorted
    observed scores; ties prefer the lower threshold (fewer false abstentions).
    """
    scores = sorted({*positives, *negatives})
    if not scores:
        return 0.0, float("nan")
    candidates = [scores[0] - 1e-6] + [(a + b) / 2 for a, b in pairwise(scores)]
    total = len(positives) + len(negatives)
    best = (candidates[0], -1.0)
    for t in candidates:
        correct = sum(p >= t for p in positives) + sum(n < t for n in negatives)
        accuracy = correct / total
        if accuracy > best[1]:
            best = (t, accuracy)
    return best
