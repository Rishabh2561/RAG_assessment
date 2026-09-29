import numpy as np
import pytest

from rag_generator.embeddings import FastEmbedProvider, HashingEmbeddingProvider


def test_hashing_is_deterministic_and_normalised():
    embedder = HashingEmbeddingProvider(256)
    a = embedder.embed_documents(["refund policy for orders", "annual leave days"])
    b = embedder.embed_documents(["refund policy for orders", "annual leave days"])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    assert a.shape == (2, 256) and a.dtype == np.float32


def test_hashing_similar_texts_score_higher():
    embedder = HashingEmbeddingProvider(1024)
    query = embedder.embed_query("how many days of annual leave")
    docs = embedder.embed_documents(
        ["employees get 25 days of annual leave", "robot payload is 150 kg"]
    )
    scores = docs @ query
    assert scores[0] > scores[1]


def test_model_id_identifies_vector_space():
    assert HashingEmbeddingProvider(128).model_id != HashingEmbeddingProvider(256).model_id
    assert (
        FastEmbedProvider("BAAI/bge-small-en-v1.5").model_id == "fastembed:BAAI/bge-small-en-v1.5"
    )


@pytest.mark.slow
def test_fastembed_real_model_semantics():
    embedder = FastEmbedProvider("BAAI/bge-small-en-v1.5")
    query = embedder.embed_query("Can I work from another country?")
    docs = embedder.embed_documents(
        ["Working temporarily from abroad is permitted for 20 days.", "The robot weighs 210 kg."]
    )
    scores = docs @ query
    assert scores[0] > scores[1]
    assert docs.shape[1] == 384
