"""Evaluation runner: retrieval metrics (no LLM needed) and answer metrics (LLM needed)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag_generator.errors import RAGError
from rag_generator.evaluation.dataset import EvalItem
from rag_generator.evaluation.metrics import (
    best_threshold,
    hit_at_k,
    keyword_recall,
    mean,
    percentile,
    reciprocal_rank,
    source_recall_at_k,
)
from rag_generator.generation import LLMProvider
from rag_generator.models import Answer
from rag_generator.orchestration.factory import RAGApplication, build_retriever
from rag_generator.reranking import Reranker

K_VALUES = (1, 3, 5, 10)

JUDGE_SYSTEM_PROMPT = """\
You check whether an answer is faithful to its cited source passages. A claim is \
supported only if the passages state it or it follows directly from them; general \
knowledge does not count. List each claim in the answer that the passages do not \
support. Set "faithful" to true only if there are none."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["faithful", "unsupported_claims"],
    "additionalProperties": False,
}


@dataclass
class RetrievalReport:
    mode: str
    reranker: str
    metrics: dict[str, float]
    by_category: dict[str, float]
    calibration: dict[str, float] | None
    items: list[dict[str, Any]] = field(default_factory=list)


class EvaluationRunner:
    def __init__(self, app: RAGApplication, collection: str) -> None:
        self.app = app
        self.collection_name = collection
        self.collection = app.repository.open(collection)

    # --- Retrieval ------------------------------------------------------------------

    def evaluate_retrieval(
        self, items: list[EvalItem], mode: str, reranker: Reranker
    ) -> RetrievalReport:
        retriever = build_retriever(self.app.settings, self.collection, self.app.embedder, mode)
        max_k = max(K_VALUES)
        pool = max(self.app.settings.candidate_pool, max_k)
        rows, latencies = [], []
        pos_scores, neg_scores = [], []
        for item in items:
            start = time.perf_counter()
            result = retriever.retrieve(item.question, pool if reranker.name != "none" else max_k)
            ranked = reranker.rerank(item.question, result.chunks, max_k)
            latencies.append((time.perf_counter() - start) * 1000)
            if result.best_dense_score is not None:
                (pos_scores if item.answerable else neg_scores).append(result.best_dense_score)
            if not item.answerable:
                continue
            relevance = [item.is_relevant(r.chunk) for r in ranked]
            sources = [r.chunk.source for r in ranked]
            rows.append(
                {
                    "id": item.id,
                    "category": item.category,
                    "first_relevant_rank": next(
                        (i + 1 for i, rel in enumerate(relevance) if rel), None
                    ),
                    "rr": reciprocal_rank(relevance),
                    **{f"hit@{k}": hit_at_k(relevance, k) for k in K_VALUES},
                    "source_recall@5": source_recall_at_k(sources, item.expected_sources, 5),
                    "best_dense": result.best_dense_score,
                }
            )
        metrics = {f"hit@{k}": mean([r[f"hit@{k}"] for r in rows]) for k in K_VALUES}
        metrics["mrr@10"] = mean([r["rr"] for r in rows])
        metrics["source_recall@5"] = mean([r["source_recall@5"] for r in rows])
        metrics["latency_p50_ms"] = percentile(latencies, 50)
        metrics["latency_p95_ms"] = percentile(latencies, 95)
        categories = sorted({r["category"] for r in rows})
        by_category = {
            c: mean([r["hit@5"] for r in rows if r["category"] == c]) for c in categories
        }
        return RetrievalReport(
            mode=mode,
            reranker=reranker.name,
            metrics=metrics,
            by_category=by_category,
            calibration=_calibration(pos_scores, neg_scores),
            items=rows,
        )

    # --- Answers --------------------------------------------------------------------

    def evaluate_answers(self, items: list[EvalItem], judge: bool = False) -> dict[str, Any]:
        service = self.app.query(self.collection_name)
        if service.llm is None:
            raise RAGError("answer evaluation needs an LLM; set RAG_LLM_PROVIDER=anthropic")
        rows: list[dict[str, Any]] = []
        for item in items:
            try:
                answer = service.ask(item.question)
            except RAGError as exc:
                rows.append({"id": item.id, "error": f"{type(exc).__name__}: {exc}"})
                continue
            rows.append(self._score_answer(item, answer, service.llm if judge else None))
        return {"metrics": _aggregate_answers(rows), "items": rows}

    def _score_answer(
        self, item: EvalItem, answer: Answer, judge: LLMProvider | None
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": item.id,
            "category": item.category,
            "expected_answerable": item.answerable,
            "predicted_answerable": answer.answerable,
            "status": answer.grounding_status,
            "invalid_citations": len(answer.trace.invalid_citations),
            "num_citations": len(answer.citations),
            "latency_ms": answer.trace.total_ms,
            "input_tokens": answer.trace.input_tokens,
            "output_tokens": answer.trace.output_tokens,
            "attempts": answer.trace.attempts,
            "answer": answer.answer,
        }
        if item.answerable and answer.answerable:
            cited_chunks = {c.chunk_id for c in answer.citations}
            passages = {p.chunk.chunk_id: p.chunk for p in answer.passages}
            row["citation_hit"] = float(
                any(item.is_relevant(passages[cid]) for cid in cited_chunks if cid in passages)
            )
            row["keyword_recall"] = keyword_recall(answer.answer or "", item.expected_keywords)
            if judge is not None:
                row["faithful"] = _judge(judge, answer)
        return row


def _judge(llm: LLMProvider, answer: Answer) -> float:
    cited = {c.chunk_id for c in answer.citations}
    passages = "\n\n".join(
        f"<passage>\n{p.chunk.text}\n</passage>"
        for p in answer.passages
        if p.chunk.chunk_id in cited
    )
    user = f"<passages>\n{passages}\n</passages>\n\n<answer>\n{answer.answer}\n</answer>"
    try:
        verdict = llm.generate_json(JUDGE_SYSTEM_PROMPT, user, JUDGE_SCHEMA).data
    except RAGError:
        return float("nan")
    return 1.0 if verdict.get("faithful") else 0.0


def _calibration(positives: list[float], negatives: list[float]) -> dict[str, float] | None:
    if not positives or not negatives:
        return None
    threshold, accuracy = best_threshold(positives, negatives)
    return {
        "answerable_best_dense_min": min(positives),
        "answerable_best_dense_p10": percentile(positives, 10),
        "unanswerable_best_dense_max": max(negatives),
        "unanswerable_best_dense_p90": percentile(negatives, 90),
        "suggested_min_relevance": round(threshold, 3),
        "gate_accuracy_at_suggested": accuracy,
    }


def _aggregate_answers(rows: list[dict[str, Any]]) -> dict[str, float]:
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["expected_answerable"]]
    unanswerable = [r for r in ok if not r["expected_answerable"]]
    answered = [r for r in answerable if r["predicted_answerable"]]

    def values(key: str) -> list[float]:
        return [r[key] for r in answered if key in r and r[key] == r[key]]  # drop NaN

    return {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "abstention_accuracy": mean(
            [float(r["expected_answerable"] == r["predicted_answerable"]) for r in ok]
        ),
        "false_abstention_rate": mean([float(not r["predicted_answerable"]) for r in answerable]),
        "false_answer_rate": mean([float(r["predicted_answerable"]) for r in unanswerable]),
        "citation_validity": mean([float(r["invalid_citations"] == 0) for r in ok]),
        "grounded_rate": mean([float(r["status"] == "grounded") for r in answered]),
        "citation_hit_rate": mean(values("citation_hit")),
        "keyword_recall": mean(values("keyword_recall")),
        "faithfulness": mean(values("faithful")),
        "latency_p50_ms": percentile([r["latency_ms"] for r in ok], 50),
        "latency_p95_ms": percentile([r["latency_ms"] for r in ok], 95),
        "avg_input_tokens": mean([r["input_tokens"] for r in ok]),
        "avg_output_tokens": mean([r["output_tokens"] for r in ok]),
    }
