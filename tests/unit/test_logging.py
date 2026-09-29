import json
import logging

from rag_generator.observability import configure_logging, content_field, get_logger, log_event
from rag_generator.observability.logging import JsonFormatter, StageTimer


def _format(record_fields: dict) -> dict:
    record = logging.LogRecord("rag_generator.x", logging.INFO, __file__, 1, "evt", None, None)
    record.fields = record_fields
    return json.loads(JsonFormatter().format(record))


def test_json_formatter_emits_structured_fields():
    payload = _format({"doc_id": "abc", "chunks": 3})
    assert payload["event"] == "evt" and payload["doc_id"] == "abc" and payload["chunks"] == 3
    assert "ts" in payload and payload["level"] == "INFO"


def test_content_is_redacted_by_default():
    configure_logging("INFO", "json", log_content=False)
    assert content_field("secret salary data") == "<redacted len=18>"
    configure_logging("INFO", "json", log_content=True)
    assert content_field("visible") == "visible"
    configure_logging("INFO", "text", log_content=False)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_log_event_attaches_fields():
    configure_logging("INFO", "text")
    handler = _ListHandler()
    root = logging.getLogger("rag_generator")
    root.addHandler(handler)
    try:
        log_event(get_logger("test"), "something_happened", doc_id="d1")
    finally:
        root.removeHandler(handler)
    assert handler.records[-1].getMessage() == "something_happened"
    assert handler.records[-1].fields == {"doc_id": "d1"}


def test_stage_timer_records_stages():
    timer = StageTimer()
    with timer.stage("a"):
        pass
    with timer.stage("b"):
        pass
    assert [name for name, _ in timer.timings] == ["a", "b"]
    assert timer.total_ms >= 0
