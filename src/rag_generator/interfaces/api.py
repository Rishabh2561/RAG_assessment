"""REST API. A thin adapter over the services; Swagger UI is served at ``/docs``."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from rag_generator import __version__
from rag_generator.config import DEFAULT_LLM_MODELS, detect_provider
from rag_generator.errors import (
    CollectionEmptyError,
    CollectionNotFoundError,
    DocumentNotFoundError,
    FileTooLargeError,
    GenerationError,
    IndexMismatchError,
    InvalidQueryError,
    RAGError,
)
from rag_generator.evaluation import RERANKERS, RETRIEVAL_MODES, parse_dataset, run_evaluation
from rag_generator.generation import LLMProvider
from rag_generator.models import Answer, DocumentRecord, IngestReport, IngestResult
from rag_generator.orchestration import RAGApplication

_STATUS_BY_ERROR: list[tuple[type[RAGError], int]] = [
    (InvalidQueryError, 400),
    (CollectionNotFoundError, 404),
    (DocumentNotFoundError, 404),
    (CollectionEmptyError, 409),
    (IndexMismatchError, 409),
    (FileTooLargeError, 413),
    (GenerationError, 503),
]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the refund policy?"])
    top_k: int | None = Field(None, ge=1, le=50)
    mode: Literal["dense", "bm25", "hybrid"] | None = None
    retrieval_only: bool = False
    include_passages: bool = False
    rewrite: bool | None = Field(None, description="Override RAG_QUERY_REWRITE for this query")


def client_llm(
    request: Request,
    x_llm_api_key: Annotated[str | None, Header(description="Use this API key")] = None,
    x_llm_provider: Annotated[
        str | None, Header(description="anthropic | openai | auto (detect from key)")
    ] = None,
    x_llm_model: Annotated[str | None, Header(description="Model for this key")] = None,
) -> tuple[str, LLMProvider] | None:
    """Request-scoped LLM from caller-supplied credentials; None = server default.

    The key is used only to build the provider for this request (cached in memory
    by key hash); it is never logged, persisted or echoed back.
    """
    rag: RAGApplication = request.app.state.rag
    key = (x_llm_api_key or "").strip()
    if not key:
        if x_llm_provider or x_llm_model:
            raise InvalidQueryError("X-LLM-Provider / X-LLM-Model require X-LLM-API-Key")
        return None
    if not rag.settings.allow_client_llm_keys:
        raise HTTPException(403, "this server does not accept client-supplied LLM keys")
    requested = (x_llm_provider or "auto").strip().lower()
    provider = detect_provider(key) if requested == "auto" else requested
    if provider is None:
        raise InvalidQueryError(
            "could not tell which provider this API key belongs to; "
            "choose Anthropic or OpenAI explicitly"
        )
    if provider not in ("anthropic", "openai"):
        raise InvalidQueryError(f"unknown LLM provider '{provider}'; use anthropic or openai")
    return provider, rag.llm_for(provider, key, (x_llm_model or "").strip() or None)


# Module level (not inside create_app): with postponed annotations FastAPI resolves
# dependency annotations by name from module globals.
ClientLLM = Annotated[tuple[str, LLMProvider] | None, Depends(client_llm)]


def create_app(rag: RAGApplication) -> FastAPI:
    api = FastAPI(
        title="RAG Generator",
        version=__version__,
        description="Upload documents into a collection, then ask grounded, cited questions.",
    )
    max_bytes = int(rag.settings.max_file_mb * 1024 * 1024)

    api.state.rag = rag

    @api.exception_handler(RAGError)
    async def _rag_error(_: Request, exc: RAGError) -> JSONResponse:
        status = next((code for cls, code in _STATUS_BY_ERROR if isinstance(exc, cls)), 500)
        body = {"error": type(exc).__name__, "detail": str(exc)}
        if isinstance(exc, GenerationError):
            body["retryable"] = exc.retryable
        return JSONResponse(status_code=status, content=body)

    @api.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        # Browsers opening the API's base URL land on the interactive docs, not a 404.
        # (The Streamlit UI is a separate process: `rag ui`, default port 8501.)
        return RedirectResponse(url="/docs")

    @api.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "embedding_model": rag.embedder.model_id,
            "llm_provider": _provider_name(rag),
            "llm_model": rag.llm.model_id if rag.llm else None,
            "retrieval_mode": rag.settings.retrieval_mode,
            "reranker": rag.reranker.name,
        }

    @api.get("/config")
    def config() -> dict:
        """What clients need to adapt to this server: formats, limits, active components.

        Never includes secrets; only whether a key is configured is implied by the provider.
        """
        s = rag.settings
        return {
            "supported_extensions": rag.registry.supported_extensions,
            "max_file_mb": s.max_file_mb,
            "max_upload_files": s.max_upload_files,
            "max_question_chars": s.max_question_chars,
            "llm_provider": _provider_name(rag),
            "llm_model": rag.llm.model_id if rag.llm else None,
            "llm_provider_auto": s.llm_provider_auto,
            "accepts_client_llm_keys": s.allow_client_llm_keys,
            "llm_providers": ["anthropic", "openai"],
            "default_llm_models": dict(DEFAULT_LLM_MODELS),
            "embedding_model": rag.embedder.model_id,
            "retrieval_modes": list(RETRIEVAL_MODES),
            "rerankers": list(RERANKERS),
            "defaults": {
                "retrieval_mode": s.retrieval_mode,
                "top_k": s.top_k,
                "reranker": rag.reranker.name,
                "query_rewrite": s.query_rewrite,
                "min_relevance": s.min_relevance,
                "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap,
            },
        }

    @api.get("/collections")
    def list_collections() -> list[str]:
        return rag.repository.list_names()

    @api.post("/collections/{collection}/documents")
    async def upload_documents(
        collection: str, files: Annotated[list[UploadFile], File(description="Documents")]
    ) -> IngestReport:
        if len(files) > rag.settings.max_upload_files:
            raise HTTPException(
                status_code=413,
                detail=f"at most {rag.settings.max_upload_files} files per request",
            )
        payloads: list[tuple[str, bytes]] = []
        oversized: list[IngestResult] = []
        for upload in files:
            name = _safe_filename(upload.filename)
            data = await upload.read(max_bytes + 1)  # bounded read: never buffer more
            if len(data) > max_bytes:
                oversized.append(
                    IngestResult(
                        source=name,
                        status="failed",
                        error_type="FileTooLargeError",
                        message=f"exceeds the {rag.settings.max_file_mb:g} MB limit",
                    )
                )
            else:
                payloads.append((name, data))
        # Parsing/embedding is CPU-bound: keep it off the event loop.
        report = await run_in_threadpool(rag.ingestion().ingest_bytes, payloads, collection)
        return report.model_copy(update={"results": report.results + oversized})

    @api.get("/collections/{collection}/documents")
    def list_documents(collection: str) -> list[DocumentRecord]:
        handle = rag.repository.open(collection)
        with handle.lock:
            return handle.catalog.list()

    @api.delete("/collections/{collection}/documents/{doc_id}")
    def delete_document(collection: str, doc_id: str) -> DocumentRecord:
        record = rag.ingestion().delete_document(collection, doc_id)
        if record is None:
            raise DocumentNotFoundError(f"document '{doc_id}' not found in '{collection}'")
        return record

    @api.delete("/collections/{collection}", status_code=204)
    def drop_collection(collection: str) -> None:
        rag.repository.drop(collection)

    @api.post("/llm/verify")
    def verify_llm(override: ClientLLM) -> dict:
        """Check that an LLM (the caller's key, or the server's) answers. Makes one tiny call."""
        provider, llm = override if override else (_provider_name(rag), rag.llm)
        if llm is None:
            return {
                "ok": False,
                "provider": "none",
                "model": None,
                "detail": "no LLM configured; enter an API key",
            }
        try:
            response = llm.generate_json(
                "Reply with the requested JSON only.",
                'Return {"ok": true}.',
                {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            )
        except GenerationError as exc:
            return {
                "ok": False,
                "provider": provider,
                "model": llm.model_id,
                "detail": str(exc),
                "retryable": exc.retryable,
            }
        return {"ok": True, "provider": provider, "model": response.model or llm.model_id}

    @api.post("/collections/{collection}/query")
    def query(collection: str, request: QueryRequest, override: ClientLLM) -> Answer:
        service = rag.query(
            collection,
            mode=request.mode,
            top_k=request.top_k,
            use_llm=not request.retrieval_only,
            rewrite=request.rewrite,
            llm=override[1] if override else None,
        )
        answer = service.ask(request.question)
        if not request.include_passages and answer.grounding_status != "retrieval_only":
            answer = answer.model_copy(update={"passages": []})
        return answer

    @api.post("/collections/{collection}/evaluate")
    async def evaluate(
        collection: str,
        dataset: Annotated[UploadFile, File(description="JSONL evaluation dataset")],
        modes: Annotated[str, Form(description="Comma-separated retrieval modes")] = "dense",
        rerankers: Annotated[str, Form(description="Comma-separated rerankers")] = "none",
        answers: Annotated[
            bool, Form(description="Also evaluate generation (uses the LLM)")
        ] = False,
        judge: Annotated[bool, Form(description="LLM-judge faithfulness")] = False,
        override: ClientLLM = None,
    ) -> dict:
        raw = await dataset.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise FileTooLargeError(f"dataset exceeds the {rag.settings.max_file_mb:g} MB limit")
        try:
            items = parse_dataset(raw.decode("utf-8-sig"), origin=dataset.filename or "dataset")
        except (UnicodeDecodeError, ValueError) as exc:
            raise InvalidQueryError(f"invalid dataset: {exc}") from exc
        return await run_in_threadpool(
            run_evaluation,
            rag,
            collection,
            items,
            modes=_split(modes),
            rerankers=_split(rerankers),
            answers=answers,
            judge=judge,
            dataset_name=dataset.filename or "",
            llm=override[1] if override else None,
        )

    return api


def _provider_name(rag: RAGApplication) -> str:
    """Reflect the LLM actually in use (one may be injected without matching settings)."""
    if rag.llm is None:
        return "none"
    return rag.settings.llm_provider if rag.settings.llm_provider != "none" else "custom"


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _safe_filename(filename: str | None) -> str:
    """Keep only the base name of an uploaded file (no client-supplied directories)."""
    if not filename:
        raise HTTPException(status_code=400, detail="uploaded file has no name")
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail=f"invalid file name '{filename}'")
    return name
