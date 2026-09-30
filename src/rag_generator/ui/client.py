"""HTTP client for the RAG Generator REST API.

The UI talks to the backend only through this class, so it runs anywhere the API is
reachable and never imports backend code. API errors (``{"error", "detail"}`` bodies)
become :class:`APIError` with a readable message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

DEFAULT_TIMEOUT_S = 300.0  # ingestion and evaluation can take a while on CPU


@dataclass
class APIError(Exception):
    """A request failed. ``status`` is None when the server could not be reached."""

    message: str
    status: int | None = None
    error: str | None = None
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


class RAGClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        """``transport`` / ``http_client`` exist for tests (e.g. FastAPI's TestClient)."""
        self.base_url = base_url.rstrip("/")
        self._http = http_client or httpx.Client(
            base_url=self.base_url, timeout=timeout_s, transport=transport
        )
        self._llm_headers: dict[str, str] = {}

    # --- LLM credentials ----------------------------------------------------------------

    def set_llm(
        self, api_key: str | None, provider: str | None = None, model: str | None = None
    ) -> None:
        """Use the caller's own key for answers (sent as X-LLM-* headers, never stored by
        the server). ``provider`` None/"auto" lets the server detect it from the key;
        an empty key reverts to the server's configured LLM."""
        key = (api_key or "").strip()
        self._llm_headers = {}
        if key:
            self._llm_headers["X-LLM-API-Key"] = key
            self._llm_headers["X-LLM-Provider"] = (provider or "auto").lower()
            if model and model.strip():
                self._llm_headers["X-LLM-Model"] = model.strip()

    @property
    def uses_own_key(self) -> bool:
        return bool(self._llm_headers)

    def verify_llm(self) -> dict[str, Any]:
        return self._request("POST", "/llm/verify", headers=self._llm_headers)

    # --- System -----------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def config(self) -> dict[str, Any]:
        return self._request("GET", "/config")

    # --- Collections and documents ------------------------------------------------------

    def collections(self) -> list[str]:
        return self._request("GET", "/collections")

    def documents(self, collection: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/collections/{_q(collection)}/documents")

    def upload(self, collection: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        multipart = [("files", (name, data)) for name, data in files]
        return self._request("POST", f"/collections/{_q(collection)}/documents", files=multipart)

    def delete_document(self, collection: str, doc_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/collections/{_q(collection)}/documents/{_q(doc_id)}")

    def drop_collection(self, collection: str) -> None:
        self._request("DELETE", f"/collections/{_q(collection)}")

    # --- Query and evaluation -----------------------------------------------------------

    def query(
        self,
        collection: str,
        question: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        retrieval_only: bool = False,
        rewrite: bool | None = None,
        include_passages: bool = True,
    ) -> dict[str, Any]:
        body = {
            "question": question,
            "mode": mode,
            "top_k": top_k,
            "retrieval_only": retrieval_only,
            "rewrite": rewrite,
            "include_passages": include_passages,
        }
        return self._request(
            "POST", f"/collections/{_q(collection)}/query", json=body, headers=self._llm_headers
        )

    def evaluate(
        self,
        collection: str,
        dataset: tuple[str, bytes],
        *,
        modes: list[str],
        rerankers: list[str],
        answers: bool = False,
        judge: bool = False,
    ) -> dict[str, Any]:
        form = {
            "modes": ",".join(modes),
            "rerankers": ",".join(rerankers),
            "answers": str(answers).lower(),
            "judge": str(judge).lower(),
        }
        return self._request(
            "POST",
            f"/collections/{_q(collection)}/evaluate",
            data=form,
            files={"dataset": dataset},
            headers=self._llm_headers,
        )

    # --- Internals ----------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise APIError(f"The API did not respond in time ({exc}).", retryable=True) from exc
        except httpx.TransportError as exc:
            raise APIError(
                f"Cannot reach the API at {self.base_url}. Start it with `rag serve`."
            ) from exc
        if response.status_code >= 400:
            raise _error_from(response)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()


def _q(segment: str) -> str:
    return quote(segment, safe="")


def _error_from(response: httpx.Response) -> APIError:
    try:
        body = response.json()
    except ValueError:
        body = {}
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list):  # FastAPI request-validation errors
        detail = "; ".join(
            f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail
        )
    return APIError(
        message=str(detail or response.text or f"HTTP {response.status_code}"),
        status=response.status_code,
        error=body.get("error") if isinstance(body, dict) else None,
        retryable=bool(body.get("retryable")) if isinstance(body, dict) else False,
    )
