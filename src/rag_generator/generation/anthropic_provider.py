"""Claude via the official Anthropic SDK (ADR-007).

This is the only module that imports ``anthropic``. It maps SDK exceptions and stop
reasons onto :class:`GenerationError` so callers never handle vendor types.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from rag_generator.errors import GenerationError
from rag_generator.generation.base import LLMResponse

SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        effort: str | None = "low",
        temperature: float | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        server_fallback: bool = True,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.temperature = temperature
        self.server_fallback = server_fallback
        self._client = client
        self._client_kwargs = {"api_key": api_key, "timeout": timeout_s, "max_retries": max_retries}

    @property
    def model_id(self) -> str:
        return self.model

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        request = self._build_request(system, user, schema)
        try:
            response = self._send(request)
        except anthropic.AuthenticationError as exc:
            raise GenerationError(
                "Anthropic authentication failed; set ANTHROPIC_API_KEY "
                "(or use RAG_LLM_PROVIDER=none for retrieval-only mode)"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise GenerationError("LLM rate limit exceeded after retries", retryable=True) from exc
        except anthropic.APITimeoutError as exc:
            raise GenerationError("LLM request timed out after retries", retryable=True) from exc
        except anthropic.APIConnectionError as exc:
            raise GenerationError("could not reach the LLM API", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            raise GenerationError(
                f"LLM API error {exc.status_code}: {exc.message}",
                retryable=exc.status_code >= 500,
            ) from exc
        except anthropic.AnthropicError as exc:  # e.g. no credentials could be resolved
            raise GenerationError(f"LLM client error: {exc}") from exc
        return self._parse(response)

    # --- Internals ------------------------------------------------------------------

    def _build_request(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        if self.temperature is not None:
            # Not a typed SDK parameter (current Opus/Sonnet models reject sampling
            # params); only sent when explicitly configured for models that accept it.
            request["extra_body"] = {"temperature": self.temperature}
        return request

    def _send(self, request: dict[str, Any]):
        client = self._get_client()
        if self.server_fallback:
            return client.beta.messages.create(
                **request, betas=[SERVER_FALLBACK_BETA], fallbacks="default"
            )
        return client.messages.create(**request)

    def _get_client(self):
        if self._client is None:
            self._client = anthropic.Anthropic(**self._client_kwargs)
        return self._client

    def _parse(self, response: Any) -> LLMResponse:
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise GenerationError("the model declined to answer this request")
        if stop_reason == "max_tokens":
            raise GenerationError("the model's output was truncated; increase RAG_LLM_MAX_TOKENS")
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GenerationError(f"model returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise GenerationError("model returned JSON that is not an object")
        usage = getattr(response, "usage", None)
        return LLMResponse(
            data=data,
            model=getattr(response, "model", self.model),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
