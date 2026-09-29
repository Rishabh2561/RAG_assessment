"""QueryService end-to-end with real retrieval and a scripted LLM."""

import pytest

from rag_generator.errors import (
    CollectionEmptyError,
    CollectionNotFoundError,
    GenerationError,
    InvalidQueryError,
)
from tests.conftest import (
    SAMPLE_DIR,
    FakeLLM,
    always_unanswerable,
    cite_first_source,
    failing_llm,
)


@pytest.fixture
def ingested(make_app):
    make_app().ingestion().ingest_paths([SAMPLE_DIR], "northwind")
    return make_app


def test_grounded_answer_with_resolved_citations(ingested):
    llm = FakeLLM(cite_first_source)
    answer = ingested(llm=llm).query("northwind").ask("How many days of annual leave?")
    assert answer.grounding_status == "grounded"
    assert answer.answerable
    citation = answer.citations[0]
    assert citation.source_id == "S1"
    assert citation.chunk_id == answer.passages[0].chunk.chunk_id
    assert citation.source == answer.passages[0].chunk.source
    assert answer.trace.attempts == 1 and answer.trace.input_tokens == 100
    assert "<question>" in llm.calls[0]["user"]


def test_pdf_citations_carry_page_numbers(ingested):
    answer = (
        ingested(llm=FakeLLM(cite_first_source))
        .query("northwind", mode="bm25")
        .ask("E-221 battery overheating")
    )
    assert answer.citations[0].source == "nw200_operator_manual.pdf"
    assert answer.citations[0].page == 3


def test_llm_abstention_is_reported(ingested):
    answer = ingested(llm=FakeLLM(always_unanswerable)).query("northwind").ask("Who is the CEO?")
    assert answer.grounding_status == "abstained"
    assert not answer.answerable and answer.citations == []
    assert answer.reason == "not in sources"


def test_invalid_citations_are_dropped_and_flagged(ingested):
    def cites_unknown(system, user, schema):
        return {"answerable": True, "answer": "Made up [S42].", "citations": ["S42"], "reason": ""}

    answer = ingested(llm=FakeLLM(cites_unknown)).query("northwind").ask("annual leave?")
    assert answer.grounding_status == "unverified"
    assert answer.citations == []
    assert answer.trace.invalid_citations == ["S42"]
    assert "[S42]" not in answer.answer


def test_relevance_gate_abstains_without_calling_llm(ingested):
    llm = FakeLLM(cite_first_source)
    service = ingested(llm=llm, min_relevance=0.99).query("northwind")
    answer = service.ask("How do I bake sourdough bread?")
    assert answer.grounding_status == "abstained"
    assert "threshold" in answer.reason
    assert llm.calls == []


def test_retrieval_only_mode(ingested):
    answer = ingested().query("northwind").ask("minimum password length")
    assert answer.grounding_status == "retrieval_only"
    assert answer.answer is None and answer.passages
    assert answer.trace.llm_model is None


def test_corrective_rewrite_retries_once(ingested):
    calls = {"answer": 0}

    def unanswerable_then_answer(system, user, schema):
        if "queries" in schema["properties"]:
            return {"queries": ["remote work outside country of employment"]}
        calls["answer"] += 1
        if calls["answer"] == 1:
            return {"answerable": False, "answer": "", "citations": [], "reason": "missing"}
        return cite_first_source(system, user, schema)

    llm = FakeLLM(unanswerable_then_answer)
    answer = ingested(llm=llm, query_rewrite=True).query("northwind").ask("work from abroad?")
    assert answer.grounding_status == "grounded"
    assert answer.trace.attempts == 2
    assert answer.trace.rewritten_queries == ["remote work outside country of employment"]
    assert len(llm.calls) == 3  # answer, rewrite, answer


def test_rewrite_is_bounded_to_one_retry(ingested):
    llm = FakeLLM(always_unanswerable)
    answer = ingested(llm=llm, query_rewrite=True).query("northwind").ask("Who is the CEO?")
    assert answer.grounding_status == "abstained"
    assert len(llm.calls) == 3


def test_rewrite_disabled_by_default(ingested):
    llm = FakeLLM(always_unanswerable)
    ingested(llm=llm).query("northwind").ask("Who is the CEO?")
    assert len(llm.calls) == 1


def test_llm_failure_raises_typed_error(ingested):
    with pytest.raises(GenerationError) as info:
        ingested(llm=FakeLLM(failing_llm)).query("northwind").ask("annual leave?")
    assert info.value.retryable


def test_schema_violating_output_raises(ingested):
    llm = FakeLLM(lambda s, u, sc: {"unexpected": "shape"})
    with pytest.raises(GenerationError, match="schema"):
        ingested(llm=llm).query("northwind").ask("annual leave?")


@pytest.mark.parametrize("question", ["", "   ", "x" * 5000])
def test_invalid_questions(ingested, question):
    with pytest.raises(InvalidQueryError):
        ingested().query("northwind").ask(question)


def test_missing_and_empty_collections(make_app):
    app = make_app()
    with pytest.raises(CollectionNotFoundError):
        app.query("nope")
    app.ingestion().ingest_bytes([("a.txt", b"")], "empty")  # fails -> collection stays empty
    with pytest.raises(CollectionEmptyError):
        app.query("empty").ask("anything?")


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_every_retrieval_mode_finds_exact_identifier(ingested, mode):
    answer = ingested().query("northwind", mode=mode).ask("What does error E-310 mean?")
    assert answer.trace.retrieval_mode == mode
    assert any("E-310" in p.chunk.text for p in answer.passages[:3])
