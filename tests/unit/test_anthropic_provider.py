"""AnthropicProvider against a mocked SDK client (no network)."""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from rag_generator.errors import GenerationError
from rag_generator.generation import ANSWER_SCHEMA
from rag_generator.generation.anthropic_provider import SERVER_FALLBACK_BETA, AnthropicProvider

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _response(payload, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(
                type="text", text=payload if isinstance(payload, str) else json.dumps(payload)
            ),
        ],
        usage=SimpleNamespace(input_tokens=321, output_tokens=45),
    )


class _Endpoint:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _client(result):
    endpoint = _Endpoint(result)
    return SimpleNamespace(messages=endpoint, beta=SimpleNamespace(messages=endpoint)), endpoint


def _provider(result, **kwargs):
    client, endpoint = _client(result)
    return AnthropicProvider("claude-opus-5-5", client=client, **kwargs), endpoint


def test_parses_structured_output_and_usage():
    payload = {"answerable": True, "answer": "x [S1]", "citations": ["S1"], "reason": ""}
    provider, endpoint = _provider(_response(payload))
    result = provider.generate_json("sys", "user", ANSWER_SCHEMA)
    assert result.data == payload
    assert (result.input_tokens, result.output_tokens) == (321, 45)
    assert endpoint.kwargs["output_config"]["format"]["schema"] == ANSWER_SCHEMA
    assert endpoint.kwargs["output_config"]["effort"] == "low"


def test_server_fallback_uses_beta_and_can_be_disabled():
    provider, endpoint = _provider(_response({"a": 1}))
    provider.generate_json("s", "u", {})
    assert endpoint.kwargs["betas"] == [SERVER_FALLBACK_BETA]
    assert endpoint.kwargs["fallbacks"] == "default"

    provider, endpoint = _provider(_response({"a": 1}), server_fallback=False)
    provider.generate_json("s", "u", {})
    assert "betas" not in endpoint.kwargs and "fallbacks" not in endpoint.kwargs


def test_temperature_only_sent_when_configured():
    provider, endpoint = _provider(_response({"a": 1}))
    provider.generate_json("s", "u", {})
    assert "extra_body" not in endpoint.kwargs
    provider, endpoint = _provider(_response({"a": 1}), temperature=0.0)
    provider.generate_json("s", "u", {})
    assert endpoint.kwargs["extra_body"] == {"temperature": 0.0}


@pytest.mark.parametrize(
    "stop_reason,match", [("refusal", "declined"), ("max_tokens", "RAG_LLM_MAX_TOKENS")]
)
def test_bad_stop_reasons_raise(stop_reason, match):
    provider, _ = _provider(_response("{}", stop_reason=stop_reason))
    with pytest.raises(GenerationError, match=match):
        provider.generate_json("s", "u", {})


def test_invalid_json_raises():
    provider, _ = _provider(_response("not json"))
    with pytest.raises(GenerationError, match="invalid JSON"):
        provider.generate_json("s", "u", {})


@pytest.mark.parametrize(
    "exc,retryable,match",
    [
        (
            anthropic.RateLimitError(
                "x", response=httpx2.Response(429, request=REQUEST), body=None
            ),
            True,
            "rate limit",
        ),
        (anthropic.APITimeoutError(request=REQUEST), True, "timed out"),
        (anthropic.APIConnectionError(request=REQUEST), True, "reach"),
        (
            anthropic.AuthenticationError(
                "x", response=httpx2.Response(401, request=REQUEST), body=None
            ),
            False,
            "ANTHROPIC_API_KEY",
        ),
        (
            anthropic.InternalServerError(
                "x", response=httpx2.Response(500, request=REQUEST), body=None
            ),
            True,
            "500",
        ),
        (
            anthropic.BadRequestError(
                "x", response=httpx2.Response(400, request=REQUEST), body=None
            ),
            False,
            "400",
        ),
    ],
)
def test_sdk_errors_map_to_generation_error(exc, retryable, match):
    provider, _ = _provider(exc)
    with pytest.raises(GenerationError, match=match) as info:
        provider.generate_json("s", "u", {})
    assert info.value.retryable is retryable


def test_missing_credentials_give_actionable_error(monkeypatch, tmp_path):
    """Uses the real SDK client: with no key it raises a bare TypeError, which must be mapped."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))  # no saved profiles
    provider = AnthropicProvider("claude-opus-5-5", max_retries=0, timeout_s=2)
    with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY") as info:
        provider.generate_json("s", "u", {})
    assert info.value.retryable is False
