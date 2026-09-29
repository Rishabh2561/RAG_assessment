"""Structured logging and stage timing.

Log records carry structured fields via ``extra={"fields": {...}}``. The JSON formatter
emits them as top-level keys; the text formatter appends them as ``key=value``.
Document and question text must only be logged through :func:`content_field`, which
redacts unless ``RAG_LOG_CONTENT=true``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

_LOGGER_ROOT = "rag_generator"
_content_logging_enabled = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {})
        suffix = " ".join(f"{k}={v}" for k, v in fields.items())
        base = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        text = f"{base} {suffix}" if suffix else base
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text


def configure_logging(level: str = "INFO", fmt: str = "text", log_content: bool = False) -> None:
    """Configure the package logger. Idempotent; logs go to stderr."""
    global _content_logging_enabled
    _content_logging_enabled = log_content

    logger = logging.getLogger(_LOGGER_ROOT)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith(_LOGGER_ROOT) else f"{_LOGGER_ROOT}.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})


def content_field(text: str, max_chars: int = 200) -> str:
    """Return text for logging only when content logging is enabled."""
    if not _content_logging_enabled:
        return f"<redacted len={len(text)}>"
    return text if len(text) <= max_chars else text[:max_chars] + "..."


class StageTimer:
    """Collects per-stage durations for a single operation."""

    def __init__(self) -> None:
        self.timings: list[tuple[str, float]] = []
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings.append((name, (time.perf_counter() - start) * 1000))

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000
