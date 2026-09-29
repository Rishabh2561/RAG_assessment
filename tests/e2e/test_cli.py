"""CLI end-to-end via typer's CliRunner, configured through environment variables only."""

import json

import pytest
from typer.testing import CliRunner

from rag_generator.interfaces.cli import app
from tests.conftest import ALT_SAMPLE_DIR, ROOT, SAMPLE_DIR

runner = CliRunner()


@pytest.fixture(autouse=True)
def offline_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no stray .env from the repo
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "none")
    monkeypatch.setenv("RAG_MIN_RELEVANCE", "0")


def test_two_document_sets_without_code_changes():
    result = runner.invoke(app, ["ingest", str(SAMPLE_DIR), "-c", "northwind"])
    assert result.exit_code == 0, result.output
    assert "0 failed" in result.output
    result = runner.invoke(app, ["ingest", str(ALT_SAMPLE_DIR), "-c", "garden"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["ask", "compost greens browns ratio", "-c", "garden", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)  # logs go to stderr, JSON to stdout
    assert payload["grounding_status"] == "retrieval_only"
    assert payload["passages"][0]["chunk"]["source"] == "composting_guide.txt"

    result = runner.invoke(app, ["collections"])
    assert "garden" in result.output and "northwind" in result.output


def test_docs_delete_and_drop():
    runner.invoke(app, ["ingest", str(ALT_SAMPLE_DIR), "-c", "g"])
    listing = runner.invoke(app, ["docs", "-c", "g"])
    doc_id = listing.output.split()[0]
    assert runner.invoke(app, ["delete", doc_id, "-c", "g"]).exit_code == 0
    assert doc_id not in runner.invoke(app, ["docs", "-c", "g"]).output
    assert runner.invoke(app, ["drop", "-c", "g", "--yes"]).exit_code == 0
    assert "No collections" in runner.invoke(app, ["collections"]).output


def test_ask_prints_trace_in_retrieval_only_mode():
    runner.invoke(app, ["ingest", str(SAMPLE_DIR), "-c", "n"])
    result = runner.invoke(app, ["ask", "E-221", "-c", "n", "--mode", "bm25", "--trace"])
    assert result.exit_code == 0
    assert "retrieval-only" in result.output and "trace: mode=bm25" in result.output


def test_errors_are_reported_cleanly():
    result = runner.invoke(app, ["ask", "anything", "-c", "missing"])
    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert "Traceback" not in result.output


def test_invalid_config_exits_with_message(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "5000")
    result = runner.invoke(app, ["collections"])
    assert result.exit_code == 2
    assert "RAG_CHUNK_OVERLAP" in result.output


def test_all_failed_batch_exits_nonzero(tmp_path):
    bad = tmp_path / "empty.txt"
    bad.write_text("")
    result = runner.invoke(app, ["ingest", str(bad), "-c", "x"])
    assert result.exit_code == 1
    assert "EmptyDocumentError" in result.output or "no text" in result.output


def test_eval_command_writes_report(tmp_path):
    runner.invoke(app, ["ingest", str(SAMPLE_DIR), "-c", "n"])
    dataset = ROOT / "data" / "eval" / "northwind_eval.jsonl"
    result = runner.invoke(
        app, ["eval", str(dataset), "-c", "n", "--modes", "bm25", "--out-dir", str(tmp_path / "r")]
    )
    assert result.exit_code == 0, result.output
    assert "hit@1" in result.output
    report = json.loads(next((tmp_path / "r").glob("*.json")).read_text())
    assert report["retrieval"][0]["mode"] == "bm25"
