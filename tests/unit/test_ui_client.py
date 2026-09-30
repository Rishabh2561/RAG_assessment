"""RAGClient against httpx.MockTransport: request shapes and error translation."""

import json

import httpx
import pytest

from rag_generator.ui.client import APIError, RAGClient


def _client(handler) -> RAGClient:
    return RAGClient("http://api.test", transport=httpx.MockTransport(handler))


def test_query_sends_all_options_and_parses_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"grounding_status": "grounded"})

    result = _client(handler).query("hr", "Leave?", mode="bm25", top_k=3, rewrite=True)
    assert result == {"grounding_status": "grounded"}
    assert seen["url"] == "http://api.test/collections/hr/query"
    assert seen["body"] == {
        "question": "Leave?",
        "mode": "bm25",
        "top_k": 3,
        "retrieval_only": False,
        "rewrite": True,
        "include_passages": True,
    }


def test_path_segments_are_url_encoded():
    seen = {}

    def handler(request):
        seen["path"] = request.url.raw_path.decode()
        return httpx.Response(200, json={})

    _client(handler).delete_document("../etc", "a/b")
    assert seen["path"] == "/collections/..%2Fetc/documents/a%2Fb"


def test_upload_is_multipart_with_file_names():
    def handler(request):
        body = request.content
        assert b'name="files"; filename="a.txt"' in body and b"hello" in body
        return httpx.Response(200, json={"collection": "c", "results": []})

    assert _client(handler).upload("c", [("a.txt", b"hello")])["collection"] == "c"


def test_evaluate_sends_dataset_and_form_fields():
    def handler(request):
        body = request.content
        assert b'filename="set.jsonl"' in body
        for field, value in [(b"modes", b"dense,bm25"), (b"answers", b"false")]:
            assert b'name="' + field + b'"' in body and value in body
        return httpx.Response(200, json={"retrieval": []})

    report = _client(handler).evaluate(
        "c",
        ("set.jsonl", b'{"id":"1","question":"q"}'),
        modes=["dense", "bm25"],
        rerankers=["none"],
    )
    assert report == {"retrieval": []}


def test_api_errors_carry_type_detail_and_retryable():
    def handler(request):
        return httpx.Response(
            503, json={"error": "GenerationError", "detail": "rate limited", "retryable": True}
        )

    with pytest.raises(APIError) as info:
        _client(handler).query("c", "q")
    assert (info.value.status, info.value.error, str(info.value)) == (
        503,
        "GenerationError",
        "rate limited",
    )
    assert info.value.retryable


def test_validation_errors_are_flattened():
    def handler(request):
        return httpx.Response(
            422, json={"detail": [{"loc": ["body", "question"], "msg": "too short"}]}
        )

    with pytest.raises(APIError, match="question: too short"):
        _client(handler).query("c", "")


def test_unreachable_server_gives_start_hint():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(APIError, match="rag serve") as info:
        _client(handler).health()
    assert info.value.status is None


def test_no_content_returns_none():
    assert _client(lambda r: httpx.Response(204)).drop_collection("c") is None
