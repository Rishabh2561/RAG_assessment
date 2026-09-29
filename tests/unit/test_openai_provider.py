"""OpenAIProvider and OpenAIEmbeddingProvider against mocked SDK clients (no network)."""

import json
from types import SimpleNamespace

import httpx2
import numpy as np
import openai
import pytest

from rag_generator.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag_generator.errors import EmbeddingError, GenerationError
from rag_generator.generation import ANSWER_SCHEMA, REWRITE_SCHEMA
from rag_generator.generation.openai_provider import OpenAIProvider

REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def _completion(content, finish_reason="stop", refusal=None):
    text = content if isinstance(content, str) else json.dumps(content)
    return SimpleNamespace(
        model="gpt-5.5-2026-04-23",
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text, refusal=refusal),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=250, completion_tokens=30),
    )


class _Endpoint:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result(kwargs) if callable(self.result) else self.result


def _llm(result, **kwargs):
    endpoint = _Endpoint(result)
    client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    return OpenAIProvider("gpt-5.5", client=client, **kwargs), endpoint


# --- Chat / structured output ------------------------------------------------------


def test_parses_structured_output_and_usage():
    payload = {"answerable": True, "answer": "x [S1]", "citations": ["S1"], "reason": ""}
    provider, endpoint = _llm(_completion(payload))
    result = provider.generate_json("system prompt", "user prompt", ANSWER_SCHEMA)
    assert result.data == payload
    assert (result.input_tokens, result.output_tokens) == (250, 30)
    assert result.model == "gpt-5.5-2026-04-23"
    request = endpoint.calls[0]
    assert request["messages"][0] == {"role": "system", "content": "system prompt"}
    assert request["messages"][1] == {"role": "user", "content": "user prompt"}
    fmt = request["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == ANSWER_SCHEMA


@pytest.mark.parametrize("schema", [ANSWER_SCHEMA, REWRITE_SCHEMA])
def test_rag_schemas_satisfy_openai_strict_mode(schema):
    """Strict mode needs additionalProperties=false and every property required."""
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_optional_parameters_only_sent_when_configured():
    provider, endpoint = _llm(_completion({"a": 1}))
    provider.generate_json("s", "u", {})
    assert "reasoning_effort" not in endpoint.calls[0]
    assert "temperature" not in endpoint.calls[0]
    provider, endpoint = _llm(_completion({"a": 1}), reasoning_effort="low", temperature=0.0)
    provider.generate_json("s", "u", {})
    assert endpoint.calls[0]["reasoning_effort"] == "low"
    assert endpoint.calls[0]["temperature"] == 0.0


@pytest.mark.parametrize(
    "response,match",
    [
        (_completion("", refusal="I can't help with that"), "declined"),
        (_completion('{"answer": "trunc', finish_reason="length"), "RAG_LLM_MAX_TOKENS"),
        (_completion("", finish_reason="content_filter"), "content filter"),
        (_completion("not json"), "invalid JSON"),
        (_completion("[1, 2]"), "not an object"),
    ],
)
def test_unusable_responses_raise(response, match):
    provider, _ = _llm(response)
    with pytest.raises(GenerationError, match=match):
        provider.generate_json("s", "u", {})


@pytest.mark.parametrize(
    "exc,retryable,match",
    [
        (
            openai.RateLimitError("x", response=httpx2.Response(429, request=REQUEST), body=None),
            True,
            "rate limit",
        ),
        (openai.APITimeoutError(request=REQUEST), True, "timed out"),
        (openai.APIConnectionError(request=REQUEST), True, "reach"),
        (
            openai.AuthenticationError(
                "x", response=httpx2.Response(401, request=REQUEST), body=None
            ),
            False,
            "OPENAI_API_KEY",
        ),
        (
            openai.InternalServerError(
                "x", response=httpx2.Response(500, request=REQUEST), body=None
            ),
            True,
            "500",
        ),
        (
            openai.BadRequestError("x", response=httpx2.Response(400, request=REQUEST), body=None),
            False,
            "400",
        ),
    ],
)
def test_sdk_errors_map_to_generation_error(exc, retryable, match):
    provider, _ = _llm(exc)
    with pytest.raises(GenerationError, match=match) as info:
        provider.generate_json("s", "u", {})
    assert info.value.retryable is retryable


def test_missing_credentials_give_actionable_error(monkeypatch):
    """Real SDK client: with no key, OpenAI() raises OpenAIError, which must be mapped."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    provider = OpenAIProvider("gpt-5.5", max_retries=0, timeout_s=2)
    with pytest.raises(GenerationError, match="OPENAI_API_KEY") as info:
        provider.generate_json("s", "u", {})
    assert info.value.retryable is False


def test_base_url_is_passed_to_client(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=_Endpoint(_completion({"a": 1})))

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    provider = OpenAIProvider("llama3", api_key="k", base_url="http://localhost:11434/v1")
    provider.generate_json("s", "u", {})
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"] == "k"


# --- Embeddings -------------------------------------------------------------------


def _embeddings_client(dim=4):
    endpoint = _Endpoint(
        lambda kwargs: SimpleNamespace(
            data=[  # deliberately returned in reverse order: must be re-sorted by index
                SimpleNamespace(index=i, embedding=[float(i + 1)] * dim)
                for i in reversed(range(len(kwargs["input"])))
            ]
        )
    )
    return SimpleNamespace(embeddings=endpoint), endpoint


def test_embeddings_batch_sort_and_normalise():
    client, endpoint = _embeddings_client()
    embedder = OpenAIEmbeddingProvider("text-embedding-3-small", batch_size=2, client=client)
    vectors = embedder.embed_documents(["a", "b", "c"])
    assert vectors.shape == (3, 4) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
    assert [len(c["input"]) for c in endpoint.calls] == [2, 1]  # batching
    assert endpoint.calls[0]["model"] == "text-embedding-3-small"
    assert embedder.model_id == "openai:text-embedding-3-small"


def test_embedding_query_vector():
    client, _ = _embeddings_client(dim=8)
    vector = OpenAIEmbeddingProvider(client=client).embed_query("question")
    assert vector.shape == (8,)


def test_embedding_errors_are_wrapped():
    endpoint = _Endpoint(openai.APIConnectionError(request=REQUEST))
    embedder = OpenAIEmbeddingProvider(client=SimpleNamespace(embeddings=endpoint))
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        embedder.embed_documents(["text"])


def test_embedding_count_mismatch_is_detected():
    endpoint = _Endpoint(SimpleNamespace(data=[]))
    embedder = OpenAIEmbeddingProvider(client=SimpleNamespace(embeddings=endpoint))
    with pytest.raises(EmbeddingError, match="expected 1"):
        embedder.embed_documents(["text"])
