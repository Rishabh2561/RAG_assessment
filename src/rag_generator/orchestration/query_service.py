"""Query pipeline (read path): retrieve → rerank → gate → generate → validate citations.

Includes the single bounded corrective retry (ADR-009): when context is judged
insufficient (by the relevance gate or by the LLM) and rewriting is enabled, the LLM
proposes alternative queries, results are fused with the original retrieval, and
generation runs once more.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from rag_generator.errors import CollectionEmptyError, GenerationError, InvalidQueryError
from rag_generator.generation import (
    ANSWER_SCHEMA,
    ANSWER_SYSTEM_PROMPT,
    NOT_FOUND_MESSAGE,
    REWRITE_SCHEMA,
    REWRITE_SYSTEM_PROMPT,
    LLMProvider,
    build_answer_prompt,
    build_rewrite_prompt,
    resolve_citations,
    strip_invalid_markers,
)
from rag_generator.models import Answer, QueryTrace, RetrievedChunk, StageTiming
from rag_generator.observability import StageTimer, content_field, get_logger, log_event
from rag_generator.reranking import Reranker
from rag_generator.retrieval import RetrievalResult, Retriever, reciprocal_rank_fusion
from rag_generator.storage import Collection

logger = get_logger(__name__)


@dataclass(frozen=True)
class QueryOptions:
    top_k: int = 5
    candidate_pool: int = 20
    min_relevance: float = 0.0
    max_context_chars: int = 12000
    max_question_chars: int = 2000
    query_rewrite: bool = False
    max_rewrites: int = 3
    rrf_k: int = 60


class _AnswerPayload(BaseModel):
    answerable: bool
    answer: str = ""
    citations: list[str] = []
    reason: str = ""


@dataclass(frozen=True)
class _Generation:
    payload: _AnswerPayload
    source_map: dict[str, RetrievedChunk]


class QueryService:
    def __init__(
        self,
        collection: Collection,
        retriever: Retriever,
        reranker: Reranker,
        llm: LLMProvider | None,
        options: QueryOptions,
    ) -> None:
        self.collection = collection
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.options = options

    def ask(self, question: str) -> Answer:
        question = self._validate(question)
        timer = StageTimer()
        trace = QueryTrace(
            collection=self.collection.name,
            retrieval_mode=self.retriever.name,
            reranker=self.reranker.name,
            llm_model=self.llm.model_id if self.llm else None,
            gate_threshold=self.options.min_relevance or None,
        )
        try:
            answer = self._run(question, trace, timer)
        except GenerationError as exc:
            self._finish(trace, timer)
            log_event(
                logger,
                "query_failed",
                level=logging.ERROR,
                error=str(exc),
                **self._log_fields(trace),
            )
            raise
        self._finish(trace, timer)
        log_event(
            logger,
            "query_completed",
            status=answer.grounding_status,
            citations=[c.chunk_id for c in answer.citations],
            question=content_field(question),
            **self._log_fields(trace),
        )
        return answer

    # --- Flow -----------------------------------------------------------------------

    def _run(self, question: str, trace: QueryTrace, timer: StageTimer) -> Answer:
        with timer.stage("retrieve"):
            result = self._retrieve(question)
        self._record_retrieval(trace, result)

        if self.llm is None:
            return self._retrieval_only(question, result, trace)

        generation = None
        if not self._gate_fails(result):
            with timer.stage("generate"):
                generation = self._generate(question, result.chunks, trace)

        if self._insufficient(generation) and self.options.query_rewrite:
            with timer.stage("rewrite"):
                queries = self._rewrite(question, trace)
            with timer.stage("retrieve_rewritten"):
                result = self._retrieve_many([question, *queries])
            self._record_retrieval(trace, result)
            if not self._gate_fails(result):
                with timer.stage("generate_retry"):
                    generation = self._generate(question, result.chunks, trace)

        if generation is None:
            return self._abstain(question, result, trace, reason=self._gate_reason(result))
        return self._build_answer(question, result, generation, trace)

    def _retrieve(self, query: str) -> RetrievalResult:
        use_reranker = self.reranker.name != "none"
        k = (
            max(self.options.candidate_pool, self.options.top_k)
            if use_reranker
            else self.options.top_k
        )
        result = self.retriever.retrieve(query, k)
        chunks = self.reranker.rerank(query, result.chunks, self.options.top_k)
        return RetrievalResult(chunks=chunks, best_dense_score=result.best_dense_score)

    def _retrieve_many(self, queries: list[str]) -> RetrievalResult:
        results = [self._retrieve(q) for q in queries]
        fused = reciprocal_rank_fusion([r.chunks for r in results], self.options.rrf_k)
        dense_scores = [r.best_dense_score for r in results if r.best_dense_score is not None]
        return RetrievalResult(
            chunks=fused[: self.options.top_k],
            best_dense_score=max(dense_scores) if dense_scores else None,
        )

    def _generate(
        self, question: str, passages: list[RetrievedChunk], trace: QueryTrace
    ) -> _Generation:
        prompt, source_map = build_answer_prompt(question, passages, self.options.max_context_chars)
        assert self.llm is not None
        response = self.llm.generate_json(ANSWER_SYSTEM_PROMPT, prompt, ANSWER_SCHEMA)
        trace.attempts += 1
        trace.input_tokens += response.input_tokens
        trace.output_tokens += response.output_tokens
        trace.llm_model = response.model
        try:
            payload = _AnswerPayload.model_validate(response.data)
        except ValidationError as exc:
            raise GenerationError(f"model output did not match the answer schema: {exc}") from exc
        return _Generation(payload=payload, source_map=source_map)

    def _rewrite(self, question: str, trace: QueryTrace) -> list[str]:
        assert self.llm is not None
        response = self.llm.generate_json(
            REWRITE_SYSTEM_PROMPT,
            build_rewrite_prompt(question, self.options.max_rewrites),
            REWRITE_SCHEMA,
        )
        trace.input_tokens += response.input_tokens
        trace.output_tokens += response.output_tokens
        raw = response.data.get("queries", [])
        queries = [q.strip() for q in raw if isinstance(q, str) and q.strip()]
        queries = [q for q in queries if q.lower() != question.lower()][: self.options.max_rewrites]
        trace.rewritten_queries = queries
        return queries

    # --- Result construction -------------------------------------------------------

    def _build_answer(
        self, question: str, result: RetrievalResult, generation: _Generation, trace: QueryTrace
    ) -> Answer:
        payload = generation.payload
        if not payload.answerable or not payload.answer.strip():
            reason = payload.reason or "The documents do not contain this information."
            return self._abstain(question, result, trace, reason=reason)

        citations, invalid = resolve_citations(
            payload.citations, payload.answer, generation.source_map
        )
        trace.invalid_citations = invalid
        if invalid:
            log_event(logger, "invalid_citations_dropped", level=logging.WARNING, ids=invalid)
        return Answer(
            question=question,
            answer=strip_invalid_markers(payload.answer, invalid).strip(),
            answerable=True,
            grounding_status="grounded" if citations else "unverified",
            citations=citations,
            passages=result.chunks,
            trace=trace,
        )

    def _abstain(
        self, question: str, result: RetrievalResult, trace: QueryTrace, reason: str
    ) -> Answer:
        return Answer(
            question=question,
            answer=NOT_FOUND_MESSAGE,
            answerable=False,
            grounding_status="abstained",
            reason=reason,
            passages=result.chunks,
            trace=trace,
        )

    def _retrieval_only(self, question: str, result: RetrievalResult, trace: QueryTrace) -> Answer:
        gated = self._gate_fails(result)
        return Answer(
            question=question,
            answer=None,
            answerable=not gated and bool(result.chunks),
            grounding_status="abstained" if gated else "retrieval_only",
            reason=self._gate_reason(result) if gated else None,
            passages=result.chunks,
            trace=trace,
        )

    # --- Helpers --------------------------------------------------------------------

    def _validate(self, question: str) -> str:
        question = (question or "").strip()
        if not question:
            raise InvalidQueryError("question must not be empty")
        if len(question) > self.options.max_question_chars:
            raise InvalidQueryError(
                f"question exceeds {self.options.max_question_chars} characters"
            )
        if len(self.collection.store) == 0:
            raise CollectionEmptyError(
                f"collection '{self.collection.name}' has no documents; ingest some first"
            )
        return question

    def _gate_fails(self, result: RetrievalResult) -> bool:
        if not result.chunks:
            return True
        threshold = self.options.min_relevance
        score = result.best_dense_score
        return threshold > 0 and score is not None and score < threshold

    def _gate_reason(self, result: RetrievalResult) -> str:
        if not result.chunks:
            return "No passages matched the question."
        return (
            f"No passage was relevant enough (best similarity {result.best_dense_score:.2f} "
            f"< threshold {self.options.min_relevance:.2f})."
        )

    @staticmethod
    def _insufficient(generation: _Generation | None) -> bool:
        return generation is None or not generation.payload.answerable

    @staticmethod
    def _record_retrieval(trace: QueryTrace, result: RetrievalResult) -> None:
        trace.retrieved_chunk_ids = [p.chunk.chunk_id for p in result.chunks]
        trace.best_dense_score = result.best_dense_score

    @staticmethod
    def _finish(trace: QueryTrace, timer: StageTimer) -> None:
        trace.timings = [StageTiming(stage=s, duration_ms=round(ms, 2)) for s, ms in timer.timings]
        trace.total_ms = round(timer.total_ms, 2)

    @staticmethod
    def _log_fields(trace: QueryTrace) -> dict:
        return {
            "collection": trace.collection,
            "mode": trace.retrieval_mode,
            "reranker": trace.reranker,
            "model": trace.llm_model,
            "attempts": trace.attempts,
            "rewrites": len(trace.rewritten_queries),
            "chunk_ids": trace.retrieved_chunk_ids,
            "best_dense": round(trace.best_dense_score, 4)
            if trace.best_dense_score is not None
            else None,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "total_ms": trace.total_ms,
            **{f"{t.stage}_ms": t.duration_ms for t in trace.timings},
        }
