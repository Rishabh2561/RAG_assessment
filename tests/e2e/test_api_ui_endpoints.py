"""API endpoints added for the UI: /config, /evaluate, and the per-query rewrite override."""

import json

import pytest
from fastapi.testclient import TestClient

from rag_generator.interfaces.api import create_app
from tests.conftest import ROOT, SAMPLE_DIR, FakeLLM, always_unanswerable


@pytest.fixture
def client_for(make_app):
    def _client(llm=None, **overrides) -> TestClient:
        return TestClient(create_app(make_app(llm=llm, **overrides)))

    return _client


def _ingest_sample(client: TestClient, name: str = "n") -> None:
    files = [("files", (p.name, p.read_bytes())) for p in SAMPLE_DIR.iterdir()]
    assert client.post(f"/collections/{name}/documents", files=files).status_code == 200


def _evaluate(client: TestClient, dataset: bytes, **form):
    return client.post(
        "/collections/n/evaluate",
        files={"dataset": ("set.jsonl", dataset)},
        data={k: str(v).lower() if isinstance(v, bool) else v for k, v in form.items()},
    )


def test_config_describes_server_without_secrets(client_for):
    body = client_for(anthropic_api_key="sk-secret").get("/config").json()
    assert {"pdf", "docx", "md", "txt"} <= set(body["supported_extensions"])
    assert body["retrieval_modes"] == ["dense", "bm25", "hybrid"]
    assert body["llm_provider"] == "none" and body["defaults"]["top_k"] == 5
    assert "sk-secret" not in json.dumps(body)


def test_evaluate_retrieval_over_http(client_for):
    client = client_for()
    _ingest_sample(client)
    dataset = (ROOT / "data" / "eval" / "northwind_eval.jsonl").read_bytes()
    response = _evaluate(client, dataset, modes="bm25,dense")
    assert response.status_code == 200, response.text
    report = response.json()
    assert [r["mode"] for r in report["retrieval"]] == ["bm25", "dense"]
    assert report["n_items"] >= 20 and report["dataset"] == "set.jsonl"
    assert "answers" not in report


@pytest.mark.parametrize(
    "dataset,form,match",
    [
        (b'{"id": "1", "question": "q"}', {"modes": "magic"}, "unknown"),
        (b"not json", {"modes": "dense"}, "invalid dataset"),
        (b"", {"modes": "dense"}, "no evaluation items"),
        (b'{"id": "1", "question": "q"}', {"modes": "dense", "answers": True}, "needs an LLM"),
    ],
)
def test_evaluate_rejects_bad_input_with_400(client_for, dataset, form, match):
    client = client_for()
    _ingest_sample(client)
    response = _evaluate(client, dataset, **form)
    assert response.status_code == 400
    assert match in response.json()["detail"]


def test_evaluate_answers_report_is_valid_json_even_with_undefined_metrics(client_for):
    client = client_for(llm=FakeLLM(always_unanswerable))
    _ingest_sample(client)
    dataset = b'{"id": "u1", "question": "Who is the CEO?", "answerable": false}'
    report = _evaluate(client, dataset, modes="dense", answers=True).json()
    metrics = report["answers"]["metrics"]
    assert metrics["false_answer_rate"] == 0.0
    assert metrics["citation_hit_rate"] is None  # NaN (no answered items) becomes null


def test_query_rewrite_can_be_enabled_per_request(client_for):
    llm = FakeLLM(always_unanswerable)
    client = client_for(llm=llm)  # server default: rewrite off
    _ingest_sample(client)
    client.post("/collections/n/query", json={"question": "Who is the CEO?"})
    assert len(llm.calls) == 1
    body = client.post("/collections/n/query", json={"question": "CEO?", "rewrite": True}).json()
    assert len(llm.calls) == 4  # + answer, rewrite, answer
    assert body["trace"]["rewritten_queries"]


def test_root_redirects_browsers_to_api_docs(client_for):
    response = client_for().get("/", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"] == "/docs"
