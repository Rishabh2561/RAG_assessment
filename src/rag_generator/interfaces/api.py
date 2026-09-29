"""REST API. A thin adapter over the services; Swagger UI is served at ``/docs``."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_generator import __version__
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


def create_app(rag: RAGApplication) -> FastAPI:
    api = FastAPI(
        title="RAG Generator",
        version=__version__,
        description="Upload documents into a collection, then ask grounded, cited questions.",
    )
    max_bytes = int(rag.settings.max_file_mb * 1024 * 1024)

    @api.exception_handler(RAGError)
    async def _rag_error(_: Request, exc: RAGError) -> JSONResponse:
        status = next((code for cls, code in _STATUS_BY_ERROR if isinstance(exc, cls)), 500)
        body = {"error": type(exc).__name__, "detail": str(exc)}
        if isinstance(exc, GenerationError):
            body["retryable"] = exc.retryable
        return JSONResponse(status_code=status, content=body)

    @api.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "embedding_model": rag.embedder.model_id,
            "llm_model": rag.llm.model_id if rag.llm else None,
            "retrieval_mode": rag.settings.retrieval_mode,
            "reranker": rag.reranker.name,
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

    @api.post("/collections/{collection}/query")
    def query(collection: str, request: QueryRequest) -> Answer:
        service = rag.query(
            collection, mode=request.mode, top_k=request.top_k, use_llm=not request.retrieval_only
        )
        answer = service.ask(request.question)
        if not request.include_passages and answer.grounding_status != "retrieval_only":
            answer = answer.model_copy(update={"passages": []})
        return answer

    return api


def _safe_filename(filename: str | None) -> str:
    """Keep only the base name of an uploaded file (no client-supplied directories)."""
    if not filename:
        raise HTTPException(status_code=400, detail="uploaded file has no name")
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail=f"invalid file name '{filename}'")
    return name
