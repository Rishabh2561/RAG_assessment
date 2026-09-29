from rag_generator.models import Chunk, RetrievedChunk
from rag_generator.retrieval import BM25Index, reciprocal_rank_fusion


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", source="s.txt", page=None, index=0, text=text)


CHUNKS = [
    _chunk("c1", "The robot shows error E-221 when the battery overheats."),
    _chunk("c2", "Battery charging takes ninety minutes on the dock."),
    _chunk("c3", "Employees receive 25 days of annual leave."),
]


def test_bm25_ranks_exact_term_first():
    hits = BM25Index(CHUNKS).search("what is E-221", k=3)
    assert hits[0][0].chunk_id == "c1"


def test_bm25_ignores_stopwords_and_unknown_terms():
    assert BM25Index(CHUNKS).search("what is the", k=3) == []
    assert BM25Index(CHUNKS).search("zebra", k=3) == []


def test_bm25_empty_corpus():
    assert BM25Index([]).search("anything", 5) == []


def _rc(cid: str, **scores) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(cid, cid), score=0.0, **scores)


def test_rrf_fuses_by_rank_and_keeps_component_scores():
    dense = [_rc("a", dense_score=0.9), _rc("b", dense_score=0.8), _rc("c", dense_score=0.7)]
    lexical = [_rc("c", lexical_score=12.0), _rc("a", lexical_score=3.0)]
    fused = reciprocal_rank_fusion([dense, lexical], rrf_k=60)
    # a: 1/61 + 1/62; c: 1/63 + 1/61; b: 1/62
    assert [f.chunk.chunk_id for f in fused] == ["a", "c", "b"]
    top = fused[0]
    assert top.dense_score == 0.9 and top.lexical_score == 3.0
    assert top.score == 1 / 61 + 1 / 62


def test_rrf_item_in_one_list_only():
    fused = reciprocal_rank_fusion([[_rc("x")], []], rrf_k=60)
    assert [f.chunk.chunk_id for f in fused] == ["x"]
