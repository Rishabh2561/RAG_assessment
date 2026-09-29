"""REST API end-to-end via FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

from rag_generator.interfaces.api import create_app
from tests.conftest import SAMPLE_DIR, FakeLLM, cite_first_source, failing_llm, make_pdf


@pytest.fixture
def client_for(make_app):
    def _client(llm=None, **overrides) -> TestClient:
        return TestClient(create_app(make_app(llm=llm, **overrides)))

    return _client


def _upload(client: TestClient, collection: str, files: list[tuple[str, bytes]]):
    return client.post(
        f"/collections/{collection}/documents",
        files=[("files", (name, data)) for name, data in files],
    )


def test_upload_then_query_with_citations(client_for):
    client = client_for(llm=FakeLLM(cite_first_source))
    files = [(p.name, p.read_bytes()) for p in SAMPLE_DIR.iterdir()]
    response = _upload(client, "northwind", files)
    assert response.status_code == 200
    assert all(r["status"] == "ingested" for r in response.json()["results"])

    response = client.post("/collections/northwind/query", json={"question": "What is E-221?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounding_status"] == "grounded"
    assert body["citations"][0]["source"]
    assert body["passages"] == []  # omitted unless requested
    assert body["trace"]["timings"]


def test_include_passages_and_retrieval_only(client_for):
    client = client_for()
    _upload(client, "c", [("m.pdf", make_pdf(["The payload limit is 150 kg for the robot."]))])
    body = client.post(
        "/collections/c/query", json={"question": "payload limit", "retrieval_only": True}
    ).json()
    assert body["grounding_status"] == "retrieval_only"
    assert body["passages"][0]["chunk"]["page"] == 1


def test_document_listing_and_deletion(client_for):
    client = client_for()
    _upload(client, "c", [("a.txt", b"Alpha document content for listing.")])
    docs = client.get("/collections/c/documents").json()
    assert [d["source"] for d in docs] == ["a.txt"]
    assert client.delete(f"/collections/c/documents/{docs[0]['doc_id']}").status_code == 200
    assert client.delete("/collections/c/documents/nope").status_code == 404
    assert client.delete("/collections/c").status_code == 204
    assert client.get("/collections").json() == []


def test_upload_filename_is_sanitised(client_for):
    client = client_for()
    response = _upload(client, "c", [("../../etc/evil.txt", b"Path traversal attempt content.")])
    assert response.json()["results"][0]["source"] == "evil.txt"


def test_oversized_upload_is_rejected_per_file(client_for):
    client = client_for(max_file_mb=0.001)
    response = _upload(
        client, "c", [("big.txt", b"x " * 2000), ("ok.txt", b"Small valid file content.")]
    )
    results = {r["source"]: r for r in response.json()["results"]}
    assert results["big.txt"]["error_type"] == "FileTooLargeError"
    assert results["ok.txt"]["status"] == "ingested"


@pytest.mark.parametrize(
    "setup,path,payload,status",
    [
        (False, "/collections/missing/query", {"question": "q"}, 404),
        (True, "/collections/c/query", {"question": ""}, 422),
        (True, "/collections/c/query", {"question": "q", "mode": "nope"}, 422),
        (False, "/collections/bad%20name/query", {"question": "q"}, 400),
    ],
)
def test_error_status_mapping(client_for, setup, path, payload, status):
    client = client_for()
    if setup:
        _upload(client, "c", [("a.txt", b"Some content to make the collection exist.")])
    assert client.post(path, json=payload).status_code == status


def test_llm_outage_returns_503_with_retryable_flag(client_for):
    client = client_for(llm=FakeLLM(failing_llm))
    _upload(client, "c", [("a.txt", b"Some content to make the collection exist.")])
    response = client.post("/collections/c/query", json={"question": "content?"})
    assert response.status_code == 503
    assert response.json()["retryable"] is True


def test_health(client_for):
    body = client_for().get("/health").json()
    assert body["status"] == "ok" and body["llm_model"] is None


def test_too_many_files_in_one_request(client_for):
    client = client_for(max_upload_files=2)
    files = [(f"f{i}.txt", b"Some small file content here.") for i in range(3)]
    assert _upload(client, "c", files).status_code == 413
