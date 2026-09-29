import math

import pytest

from rag_generator.evaluation.metrics import (
    best_threshold,
    hit_at_k,
    keyword_recall,
    mean,
    percentile,
    reciprocal_rank,
    source_recall_at_k,
)


def test_hit_at_k():
    rel = [False, False, True, False]
    assert hit_at_k(rel, 2) == 0.0 and hit_at_k(rel, 3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank([False, True]) == 0.5
    assert reciprocal_rank([False, False]) == 0.0


def test_source_recall():
    assert source_recall_at_k(["a", "b", "a"], ["a", "c"], 3) == 0.5
    assert source_recall_at_k(["a"], [], 3) == 1.0


def test_keyword_recall_case_insensitive():
    assert keyword_recall(
        "BitLocker and FileVault", ["bitlocker", "filevault", "luks"]
    ) == pytest.approx(2 / 3)


def test_percentile_and_mean():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 95) == 50
    assert mean([1, 2, 3]) == 2
    assert math.isnan(mean([]))


def test_best_threshold_separates_when_possible():
    threshold, accuracy = best_threshold([0.8, 0.7, 0.75], [0.4, 0.5])
    assert 0.5 < threshold < 0.7 and accuracy == 1.0


def test_best_threshold_with_overlap_prefers_lower_threshold():
    threshold, accuracy = best_threshold([0.6, 0.9], [0.7, 0.3])
    assert accuracy == 0.75
    assert threshold < 0.6  # keeps all answerable questions rather than blocking 0.6
