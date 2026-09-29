"""LLM provider interface.

Providers expose one capability: produce a JSON object that conforms to a schema. The
prompts, the schema and every RAG-specific rule live outside the provider, so swapping
vendors never touches grounding logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMResponse:
    data: dict[str, Any]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        """Return a JSON object matching ``schema``.

        Raises :class:`~rag_generator.errors.GenerationError` on any failure; the
        ``retryable`` flag distinguishes transient faults (rate limit, timeout, 5xx).
        """
        ...
