import numpy as np
import pytest

from rag_generator.errors import IndexMismatchError
from rag_generator.models import Chunk
from rag_generator.storage import CollectionRepository, NumpyVectorStore


def _chunk(doc_id: str, i: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-{i}",
        doc_id=doc_id,
        source=f"{doc_id}.txt",
        page=None,
        index=i,
        text=f"text {doc_id} {i}",
    )


def _unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path) -> NumpyVectorStore:
    s = NumpyVectorStore(tmp_path / "c", expected_model_id="m1")
    s.add(
        [_chunk("a", 0), _chunk("a", 1), _chunk("b", 0)],
        np.stack([_unit(1, 0, 0), _unit(0.9, 0.1, 0), _unit(0, 1, 0)]),
    )
    return s


def test_search_orders_by_cosine(store):
    hits = store.search(_unit(1, 0, 0), k=2)
    assert [c.chunk_id for c, _ in hits] == ["a-0", "a-1"]
    assert hits[0][1] == pytest.approx(1.0)


def test_k_larger_than_store(store):
    assert len(store.search(_unit(1, 0, 0), k=50)) == 3


def test_delete_document(store):
    version = store.version
    assert store.delete_document("a") == 2
    assert [c.chunk_id for c in store.all_chunks()] == ["b-0"]
    assert store.version > version
    assert store.delete_document("missing") == 0


def test_persist_and_reload_roundtrip(store, tmp_path):
    store.persist()
    reloaded = NumpyVectorStore(tmp_path / "c", expected_model_id="m1")
    assert len(reloaded) == 3
    assert reloaded.search(_unit(0, 1, 0), 1)[0][0].chunk_id == "b-0"
    assert not list((tmp_path / "c").glob("*.tmp*"))  # atomic write left no temp files


def test_model_mismatch_is_refused(store, tmp_path):
    store.persist()
    other = NumpyVectorStore(tmp_path / "c", expected_model_id="m2")
    with pytest.raises(IndexMismatchError, match="Re-ingest"):
        other.search(_unit(1, 0, 0), 1)
    with pytest.raises(IndexMismatchError):
        other.add([_chunk("c", 0)], np.stack([_unit(1, 0, 0)]))


def test_dimension_mismatch_is_refused(store):
    with pytest.raises(IndexMismatchError, match="dimension"):
        store.search(np.ones(5, dtype=np.float32), 1)


def test_empty_store_search_returns_nothing(tmp_path):
    assert NumpyVectorStore(tmp_path / "e", expected_model_id="m").search(_unit(1, 0), 3) == []


def test_repository_drops_orphan_chunks(tmp_path, store):
    """Simulates a crash after the index was written but before the catalog was."""
    store.persist()  # catalog for collection 'c' does not exist -> all chunks are orphans
    repo = CollectionRepository(tmp_path, lambda d: NumpyVectorStore(d, expected_model_id="m1"))
    assert len(repo.open("c").store) == 0
