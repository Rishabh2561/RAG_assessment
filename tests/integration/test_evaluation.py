import math

from rag_generator.evaluation import EvalItem, EvaluationRunner, load_dataset
from rag_generator.reranking import NoOpReranker
from tests.conftest import ROOT, SAMPLE_DIR, FakeLLM, cite_first_source, dump_jsonl


def test_sample_dataset_is_valid_and_labels_match_corpus(make_app):
    items = load_dataset(ROOT / "data" / "eval" / "northwind_eval.jsonl")
    assert len(items) >= 20
    assert any(not i.answerable for i in items)
    app = make_app()
    app.ingestion().ingest_paths([SAMPLE_DIR], "c")
    chunks = app.repository.open("c").store.all_chunks()
    for item in items:
        if item.answerable:  # every label must be satisfiable by at least one chunk
            assert any(item.is_relevant(c) for c in chunks), item.id


def test_retrieval_report_metrics(make_app, tmp_path):
    app = make_app()
    app.ingestion().ingest_paths([SAMPLE_DIR], "c")
    dataset = dump_jsonl(
        tmp_path / "d.jsonl",
        [
            {
                "id": "a",
                "question": "E-221 battery overheating",
                "expected_sources": ["nw200_operator_manual.pdf"],
                "expected_evidence": ["E-221"],
            },
            {"id": "b", "question": "zzz qqq", "answerable": False},
        ],
    )
    report = EvaluationRunner(app, "c").evaluate_retrieval(
        load_dataset(dataset), "bm25", NoOpReranker()
    )
    assert report.metrics["hit@3"] == 1.0
    assert report.metrics["mrr@10"] > 0
    assert report.calibration is None  # bm25 has no dense scores


def test_answer_evaluation_with_fake_llm(make_app):
    app = make_app(llm=FakeLLM(cite_first_source))
    app.ingestion().ingest_paths([SAMPLE_DIR], "c")
    items = [
        EvalItem(id="a", question="What is E-221?", expected_sources=["nw200_operator_manual.pdf"]),
        EvalItem(id="b", question="Who is the CEO?", answerable=False),
    ]
    result = EvaluationRunner(app, "c").evaluate_answers(items)
    m = result["metrics"]
    assert m["n"] == 2 and m["errors"] == 0
    assert m["citation_validity"] == 1.0
    assert m["false_answer_rate"] == 1.0  # the fake always answers: the metric must catch it
    assert m["abstention_accuracy"] == 0.5
    assert math.isnan(m["faithfulness"])  # judge not requested
