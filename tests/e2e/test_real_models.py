"""Opt-in regression test with the real local embedding model: `pytest -m slow`.

Guards the retrieval quality that DECISION_LOG D8-D10 were based on.
"""

import pytest

from rag_generator.config import Settings
from rag_generator.evaluation import EvaluationRunner, load_dataset
from rag_generator.orchestration import RAGApplication
from rag_generator.reranking import NoOpReranker
from tests.conftest import ROOT, SAMPLE_DIR


@pytest.mark.slow
def test_default_configuration_retrieval_quality(tmp_path):
    settings = Settings(data_dir=tmp_path, llm_provider="none", _env_file=None)
    app = RAGApplication(settings)
    app.ingestion().ingest_paths([SAMPLE_DIR], "n")
    items = load_dataset(ROOT / "data" / "eval" / "northwind_eval.jsonl")
    report = EvaluationRunner(app, "n").evaluate_retrieval(
        items, settings.retrieval_mode, NoOpReranker()
    )
    assert report.metrics["hit@5"] >= 0.95
    assert report.metrics["mrr@10"] >= 0.85
    # The gate must never block an answerable question at the default threshold.
    assert report.calibration["answerable_best_dense_min"] > settings.min_relevance
