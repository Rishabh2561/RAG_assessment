"""Caller-supplied LLM keys (X-LLM-* headers): the provider follows whichever key is sent."""

import logging

import pytest
from fastapi.testclient import TestClient

import rag_generator.orchestration.factory as factory
from rag_generator.errors import GenerationError
from rag_generator.generation.anthropic_provider import AnthropicProvider
from rag_generator.generation.openai_provider import OpenAIProvider
from rag_generator.interfaces.api import create_app
from tests.conftest import SAMPLE_DIR, FakeLLM, cite_first_source

ANTHROPIC_KEY = "sk-ant-api03-test-secret"
OPENAI_KEY = "sk-proj-test-secret"


@pytest.fixture
def built(monkeypatch):
    """Replace real SDK providers with fakes and record what the factory was asked for."""
    calls: list[dict] = []

    def fake_build_llm(settings, *, provider=None, api_key=None, model=None):
        if provider is None:  # the server's own default LLM: none in these tests
            return None
        calls.append({"provider": provider, "api_key": api_key, "model": model})
        llm = FakeLLM(cite_first_source)
        llm.provider = provider
        return llm

    monkeypatch.setattr(factory, "build_llm", fake_build_llm)
    return calls


@pytest.fixture
def api(make_app):
    def _api(**overrides) -> TestClient:
        client = TestClient(create_app(make_app(**overrides)))  # server has no LLM of its own
        files = [("files", (p.name, p.read_bytes())) for p in SAMPLE_DIR.iterdir()]
        client.post("/collections/n/documents", files=files)
        return client

    return _api


def _ask(client, headers=None, question="What is the minimum password length?"):
    return client.post("/collections/n/query", json={"question": question}, headers=headers or {})


@pytest.mark.parametrize("key,provider", [(ANTHROPIC_KEY, "anthropic"), (OPENAI_KEY, "openai")])
def test_provider_is_detected_from_the_key(api, built, key, provider):
    response = _ask(api(), {"X-LLM-API-Key": key})
    assert response.status_code == 200, response.text
    assert response.json()["grounding_status"] == "grounded"  # the key's LLM answered
    assert built == [{"provider": provider, "api_key": key, "model": None}]


def test_without_a_key_the_server_default_is_used(api, built):
    body = _ask(api()).json()
    assert body["grounding_status"] == "retrieval_only"  # server has no LLM configured
    assert built == []


def test_explicit_provider_and_model_override_detection(api, built):
    headers = {
        "X-LLM-API-Key": "custom-key",
        "X-LLM-Provider": "OpenAI",
        "X-LLM-Model": "gpt-5.4-mini",
    }
    assert _ask(api(), headers).status_code == 200
    assert built == [{"provider": "openai", "api_key": "custom-key", "model": "gpt-5.4-mini"}]


@pytest.mark.parametrize(
    "headers,status,match",
    [
        ({"X-LLM-API-Key": "not-a-known-format"}, 400, "choose Anthropic or OpenAI"),
        ({"X-LLM-Provider": "openai"}, 400, "require X-LLM-API-Key"),
        ({"X-LLM-API-Key": OPENAI_KEY, "X-LLM-Provider": "gemini"}, 400, "unknown LLM provider"),
    ],
)
def test_bad_credentials_headers(api, built, headers, status, match):
    response = _ask(api(), headers)
    assert response.status_code == status
    assert match in response.json()["detail"]


def test_client_keys_can_be_disabled(api, built):
    response = _ask(api(allow_client_llm_keys=False), {"X-LLM-API-Key": OPENAI_KEY})
    assert response.status_code == 403
    assert built == []


def test_providers_are_cached_per_key_and_model(api, built):
    client = api()
    for _ in range(3):
        _ask(client, {"X-LLM-API-Key": OPENAI_KEY})
    _ask(client, {"X-LLM-API-Key": OPENAI_KEY, "X-LLM-Model": "gpt-5.4-mini"})
    _ask(client, {"X-LLM-API-Key": ANTHROPIC_KEY})
    assert [c["provider"] for c in built] == ["openai", "openai", "anthropic"]


def test_verify_reports_success_failure_and_missing_llm(api, built, monkeypatch):
    client = api()
    ok = client.post("/llm/verify", headers={"X-LLM-API-Key": ANTHROPIC_KEY}).json()
    assert ok["ok"] is True and ok["provider"] == "anthropic"

    def failing(settings, *, provider=None, api_key=None, model=None):
        def fail(*_):
            raise GenerationError("Anthropic authentication failed; set ANTHROPIC_API_KEY")

        return FakeLLM(fail)

    monkeypatch.setattr(factory, "build_llm", failing)
    bad = client.post("/llm/verify", headers={"X-LLM-API-Key": "sk-ant-other"}).json()
    assert bad["ok"] is False and "authentication failed" in bad["detail"]
    none = client.post("/llm/verify").json()
    assert none["ok"] is False and "enter an API key" in none["detail"]


def test_evaluate_answers_with_client_key(api, built):
    dataset = (
        b'{"id": "q", "question": "What is the minimum password length?", '
        b'"expected_sources": ["it_security_policy.txt"]}'
    )
    response = api().post(
        "/collections/n/evaluate",
        files={"dataset": ("d.jsonl", dataset)},
        data={"modes": "dense", "answers": "true"},
        headers={"X-LLM-API-Key": OPENAI_KEY},
    )
    assert response.status_code == 200, response.text
    assert response.json()["answers"]["metrics"]["n"] == 1
    assert built[0]["provider"] == "openai"


def test_key_never_appears_in_responses_or_logs(api, built):
    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(self.format(record) + str(getattr(record, "fields", "")))

    handler = Capture()
    logging.getLogger("rag_generator").addHandler(handler)
    try:
        client = api()
        body = _ask(client, {"X-LLM-API-Key": OPENAI_KEY}).text
        body += client.get("/config").text + client.get("/health").text
        body += client.post("/llm/verify", headers={"X-LLM-API-Key": OPENAI_KEY}).text
    finally:
        logging.getLogger("rag_generator").removeHandler(handler)
    assert OPENAI_KEY not in body
    assert not any(OPENAI_KEY in r for r in records)


def test_real_providers_are_built_from_client_keys(make_app):
    """No monkeypatching: llm_for builds the real SDK-backed providers (no network)."""
    app = make_app()
    anthropic = app.llm_for("anthropic", ANTHROPIC_KEY)
    openai = app.llm_for("openai", OPENAI_KEY, "gpt-5.4-mini")
    assert isinstance(anthropic, AnthropicProvider) and anthropic.model_id == "claude-opus-5-5"
    assert anthropic._client_kwargs["api_key"] == ANTHROPIC_KEY
    assert isinstance(openai, OpenAIProvider) and openai.model_id == "gpt-5.4-mini"
    assert app.llm_for("openai", OPENAI_KEY, "gpt-5.4-mini") is openai  # cached
