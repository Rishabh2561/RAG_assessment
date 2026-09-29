"""Dense, BM25 and hybrid (reciprocal-rank fusion) retrievers (ADR-005)."""

from __future__ import annotations

from rag_generator.embeddings import EmbeddingProvider
from rag_generator.models import RetrievedChunk
from rag_generator.retrieval.base import RetrievalResult
from rag_generator.retrieval.bm25 import BM25Index
from rag_generator.storage import Collection


class DenseRetriever:
    name = "dense"

    def __init__(self, collection: Collection, embedder: EmbeddingProvider) -> None:
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        query_vector = self.embedder.embed_query(query)
        hits = self.collection.store.search(query_vector, k)
        chunks = [RetrievedChunk(chunk=c, score=s, dense_score=s) for c, s in hits]
        return RetrievalResult(chunks=chunks, best_dense_score=hits[0][1] if hits else None)


class BM25Retriever:
    name = "bm25"

    def __init__(self, collection: Collection) -> None:
        self.collection = collection

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        index: BM25Index = self.collection.derived(
            "bm25", lambda: BM25Index(self.collection.store.all_chunks())
        )
        hits = index.search(query, k)
        chunks = [RetrievedChunk(chunk=c, score=s, lexical_score=s) for c, s in hits]
        return RetrievalResult(chunks=chunks, best_dense_score=None)


class HybridRetriever:
    """Fuses dense and BM25 rankings with reciprocal-rank fusion.

    RRF score = sum over rankings of 1 / (rrf_k + rank). It uses ranks only, so the
    incomparable score scales of cosine and BM25 need no normalisation or weighting.
    """

    name = "hybrid"

    def __init__(
        self, dense: DenseRetriever, lexical: BM25Retriever, rrf_k: int = 60, pool: int = 20
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.pool = pool

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        pool = max(k, self.pool)
        dense = self.dense.retrieve(query, pool)
        lexical = self.lexical.retrieve(query, pool)
        fused = reciprocal_rank_fusion([dense.chunks, lexical.chunks], self.rrf_k)
        return RetrievalResult(chunks=fused[:k], best_dense_score=dense.best_dense_score)


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], rrf_k: int
) -> list[RetrievedChunk]:
    """Merge ranked lists; per-source scores are carried over onto the fused item."""
    fused_scores: dict[str, float] = {}
    merged: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            existing = merged.get(chunk_id)
            merged[chunk_id] = item if existing is None else _merge_scores(existing, item)
    order = sorted(fused_scores, key=lambda cid: (-fused_scores[cid], cid))
    return [merged[cid].model_copy(update={"score": fused_scores[cid]}) for cid in order]


def _merge_scores(a: RetrievedChunk, b: RetrievedChunk) -> RetrievedChunk:
    return a.model_copy(
        update={
            "dense_score": a.dense_score if a.dense_score is not None else b.dense_score,
            "lexical_score": a.lexical_score if a.lexical_score is not None else b.lexical_score,
        }
    )
