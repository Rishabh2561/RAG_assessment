"""OpenAI chat models via the official ``openai`` SDK (ADR-014).

Uses Chat Completions with strict JSON-schema structured outputs, so the RAG prompts,
schemas and citation logic are shared unchanged with the Anthropic provider. This is
the only generation module that imports ``openai``; SDK exceptions and finish reasons
are mapped onto :class:`GenerationError`.
"""

from __future__ import annotations

import json
from typing import Any

import openai

from rag_generator.errors import GenerationError
from rag_generator.generation.base import LLMResponse

_NO_CREDENTIALS = (
    "no OpenAI credentials found; set OPENAI_API_KEY (environment or .env), "
    "or run retrieval-only with --no-llm / RAG_LLM_PROVIDER=none"
)


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self._client = client
        self._client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout_s,
            "max_retries": max_retries,
        }

    @property
    def model_id(self) -> str:
        return self.model

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        try:
            response = self._get_client().chat.completions.create(
                **self._build_request(system, user, schema)
            )
        except openai.AuthenticationError as exc:
            raise GenerationError(
                "OpenAI authentication failed; check OPENAI_API_KEY "
                "(or use RAG_LLM_PROVIDER=none for retrieval-only mode)"
            ) from exc
        except openai.RateLimitError as exc:
            raise GenerationError("LLM rate limit exceeded after retries", retryable=True) from exc
        except openai.APITimeoutError as exc:
            raise GenerationError("LLM request timed out after retries", retryable=True) from exc
        except openai.APIConnectionError as exc:
            raise GenerationError("could not reach the LLM API", retryable=True) from exc
        except openai.APIStatusError as exc:
            raise GenerationError(
                f"LLM API error {exc.status_code}: {exc.message}",
                retryable=exc.status_code >= 500,
            ) from exc
        except openai.OpenAIError as exc:  # includes missing credentials at client creation
            if "credentials" in str(exc).lower() or "api_key" in str(exc):
                raise GenerationError(_NO_CREDENTIALS) from exc
            raise GenerationError(f"LLM client error: {exc}") from exc
        return self._parse(response)

    # --- Internals ------------------------------------------------------------------

    def _build_request(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "rag_response", "schema": schema, "strict": True},
            },
            "max_completion_tokens": self.max_tokens,
        }
        # Both are model-dependent (reasoning models reject temperature; non-reasoning
        # models reject reasoning_effort), so each is sent only when configured.
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            request["temperature"] = self.temperature
        return request

    def _get_client(self):
        if self._client is None:
            self._client = openai.OpenAI(**self._client_kwargs)
        return self._client

    def _parse(self, response: Any) -> LLMResponse:
        if not response.choices:
            raise GenerationError("the model returned no choices")
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise GenerationError("the model declined to answer this request")
        if choice.finish_reason == "length":
            raise GenerationError("the model's output was truncated; increase RAG_LLM_MAX_TOKENS")
        if choice.finish_reason == "content_filter":
            raise GenerationError("the response was blocked by the provider's content filter")
        try:
            data = json.loads(choice.message.content or "")
        except json.JSONDecodeError as exc:
            raise GenerationError(f"model returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise GenerationError("model returned JSON that is not an object")
        usage = getattr(response, "usage", None)
        return LLMResponse(
            data=data,
            model=getattr(response, "model", None) or self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
