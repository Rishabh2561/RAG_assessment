"""Regression tests for defects found in the independent code review (DECISION_LOG D12)."""

from pathlib import Path

import pytest

from rag_generator.errors import CollectionNotFoundError, GenerationError
from rag_generator.ingestion import ParserRegistry, TextParser
from tests.conftest import SAMPLE_DIR, FakeLLM, always_unanswerable


def _texts(app, name="c"):
    return sorted(c.text for c in app.repository.open(name).store.all_chunks())


def test_failed_replace_keeps_previous_version(make_app):
    make_app().ingestion().ingest_bytes([("b.txt", b"Original version of document b.")], "c")
    other = make_app(hashing_dimensions=512)  # a different vector space -> add() will fail
    report = other.ingestion().ingest_bytes([("b.txt", b"New version of document b.")], "c")
    assert report.results[0].error_type == "IndexMismatchError"
    collection = other.repository.open("c")
    assert collection.catalog.find_by_source("b.txt") is not None
    assert [c.text for c in collection.store.all_chunks()] == ["Original version of document b."]


def test_ghost_catalog_entry_is_reconciled_on_open(make_app):
    app = make_app()
    app.ingestion().ingest_bytes([("a.txt", b"Document that will be half-deleted.")], "c")
    collection = app.repository.open("c")
    doc_id = collection.catalog.list()[0].doc_id
    collection.store.delete_document(doc_id)
    collection.store.persist()  # simulate a crash before the catalog was written
    fresh = make_app()
    assert len(fresh.repository.open("c").catalog) == 0
    report = fresh.ingestion().ingest_bytes(
        [("a.txt", b"Document that will be half-deleted.")], "c"
    )
    assert report.results[0].status == "ingested"


def test_same_base_name_in_different_directories_stays_distinct(make_app, tmp_path):
    for sub, text in [("a", "Alpha notes content here."), ("b", "Beta notes content here.")]:
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "notes.md").write_text(text)
    app = make_app()
    report = app.ingestion().ingest_paths([tmp_path / "a", tmp_path / "b"], "c")
    assert sorted(r.source for r in report.results) == ["a/notes.md", "b/notes.md"]
    assert {r.status for r in report.results} == {"ingested"}
    report = app.ingestion().ingest_paths(
        [tmp_path / "a" / "notes.md", tmp_path / "b" / "notes.md"], "d"
    )
    assert len(app.repository.open("d").catalog) == 2


def test_single_directory_keeps_plain_relative_names(make_app):
    report = make_app().ingestion().ingest_paths([SAMPLE_DIR], "c")
    assert "employee_handbook.md" in {r.source for r in report.results}


def test_duplicate_names_within_one_upload_batch(make_app):
    report = (
        make_app()
        .ingestion()
        .ingest_bytes(
            [("x.txt", b"First file with this name."), ("x.txt", b"Second file, same name.")], "c"
        )
    )
    assert [r.status for r in report.results] == ["ingested", "failed"]
    assert report.results[1].error_type == "DuplicateSourceError"


def test_renamed_content_does_not_leave_stale_version(make_app):
    app = make_app()
    service = app.ingestion()
    service.ingest_bytes(
        [
            ("a.txt", b"Version one of the text content."),
            ("b.txt", b"Version two of the text content."),
        ],
        "c",
    )
    report = service.ingest_bytes(
        [("a.txt", b"Version two of the text content.")], "c"
    )  # a.txt now == b.txt
    assert report.results[0].status == "skipped_duplicate"
    assert "same content as b.txt" in report.results[0].message
    assert _texts(app) == ["Version two of the text content."]
    assert make_app().repository.open("c").catalog.find_by_source("a.txt") is None  # persisted


def test_unexpected_parser_exception_is_isolated(make_app):
    class ExplodingParser(TextParser):
        extensions = ("boom",)

        def parse(self, data, source):
            raise RuntimeError("library bug")

    app = make_app()
    app.registry = ParserRegistry([TextParser(), ExplodingParser()])
    report = app.ingestion().ingest_bytes(
        [("bad.boom", b"x"), ("good.txt", b"Good content survives the bad file.")], "c"
    )
    assert [r.status for r in report.results] == ["failed", "ingested"]
    assert report.results[0].error_type == "RuntimeError"
    assert len(make_app().repository.open("c").catalog) == 1  # batch was persisted


def test_bm25_cache_is_not_stale_after_concurrent_mutation(make_app):
    app = make_app()
    app.ingestion().ingest_bytes([("a.txt", b"Alpha document about zebras.")], "c")
    collection = app.repository.open("c")

    def build_while_mutating():
        chunks = collection.store.all_chunks()
        collection.store.delete_document(chunks[0].doc_id)  # a writer mutates mid-build
        return "stale-index"

    assert collection.derived("k", build_while_mutating) == "stale-index"
    assert collection.derived("k", lambda: "fresh-index") == "fresh-index"


def test_dropped_collection_refuses_writes_from_stale_handle(make_app):
    app = make_app()
    app.ingestion().ingest_bytes([("a.txt", b"Some content in the collection.")], "c")
    stale = app.repository.open("c")
    app.repository.drop("c")
    with pytest.raises(CollectionNotFoundError):
        stale.persist()
    assert not Path(stale.directory).exists()


def test_rewrite_failure_falls_back_to_abstention(make_app):
    make_app().ingestion().ingest_paths([SAMPLE_DIR], "n")

    def rewrite_fails(system, user, schema):
        if "queries" in schema["properties"]:
            raise GenerationError("LLM rate limit exceeded after retries", retryable=True)
        return always_unanswerable(system, user, schema)

    answer = make_app(llm=FakeLLM(rewrite_fails), query_rewrite=True).query("n").ask("CEO?")
    assert answer.grounding_status == "abstained"
    assert "rate limit" in answer.trace.rewrite_error


def test_gate_failure_triggers_rewrite(make_app):
    make_app().ingestion().ingest_paths([SAMPLE_DIR], "n")
    llm = FakeLLM(always_unanswerable)
    app = make_app(llm=llm, query_rewrite=True, min_relevance=0.99)
    answer = app.query("n").ask("sourdough bread")
    assert answer.grounding_status == "abstained"
    assert answer.trace.rewritten_queries  # rewrite was attempted
    assert answer.trace.attempts == 0  # gate still failed: no answer generation was paid for


def test_empty_rewrite_skips_retry(make_app):
    make_app().ingestion().ingest_paths([SAMPLE_DIR], "n")

    def no_queries(system, user, schema):
        if "queries" in schema["properties"]:
            return {"queries": []}
        return always_unanswerable(system, user, schema)

    llm = FakeLLM(no_queries)
    make_app(llm=llm, query_rewrite=True).query("n").ask("CEO?")
    assert len(llm.calls) == 2  # answer + rewrite, no pointless second answer


def test_schema_error_message_does_not_include_model_output(make_app):
    make_app().ingestion().ingest_paths([SAMPLE_DIR], "n")
    llm = FakeLLM(lambda s, u, sc: {"answerable": "SECRET-DOCUMENT-TEXT"})
    with pytest.raises(GenerationError) as info:
        make_app(llm=llm).query("n").ask("anything?")
    assert "SECRET-DOCUMENT-TEXT" not in str(info.value)
