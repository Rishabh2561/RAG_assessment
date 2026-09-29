import pytest
from pydantic import ValidationError

from rag_generator.config import Settings


def test_defaults_are_valid():
    s = Settings(_env_file=None)
    assert s.chunk_overlap < s.chunk_size
    assert s.top_k <= s.candidate_pool
    assert s.llm_model  # not hard-coded elsewhere; configurable here


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "900")
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "bm25")
    monkeypatch.setenv("RAG_LLM_MODEL", "claude-haiku-4-5")
    s = Settings(_env_file=None)
    assert (s.chunk_size, s.retrieval_mode, s.llm_model) == (900, "bm25", "claude-haiku-4-5")


def test_dotenv_file_is_read(tmp_path):
    env = tmp_path / ".env"
    env.write_text("RAG_TOP_K=7\nANTHROPIC_API_KEY=sk-test-123\n")
    s = Settings(_env_file=env)
    assert s.top_k == 7
    assert s.anthropic_api_key.get_secret_value() == "sk-test-123"


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValidationError, match="RAG_CHUNK_OVERLAP"):
        Settings(chunk_size=200, chunk_overlap=200, _env_file=None)


def test_top_k_cannot_exceed_candidate_pool():
    with pytest.raises(ValidationError, match="RAG_TOP_K"):
        Settings(top_k=30, candidate_pool=10, _env_file=None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("retrieval_mode", "magic"),
        ("llm_provider", "unknown"),
        ("embedding_provider", "x"),
        ("default_collection", "../etc"),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value}, _env_file=None)


def test_api_key_is_never_shown_in_repr(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    s = Settings(_env_file=None)
    assert "sk-super-secret" not in repr(s)
    assert "sk-super-secret" not in s.model_dump_json()


@pytest.mark.parametrize(
    "overrides,llm_model,embedding_model,gate",
    [
        ({}, "claude-opus-5-5", "BAAI/bge-small-en-v1.5", 0.50),
        ({"llm_provider": "openai"}, "gpt-5.5", "BAAI/bge-small-en-v1.5", 0.50),
        ({"embedding_provider": "openai"}, "claude-opus-5-5", "text-embedding-3-small", 0.0),
        ({"llm_provider": "none"}, None, "BAAI/bge-small-en-v1.5", 0.50),
        ({"llm_provider": "openai", "llm_model": "gpt-5.4-mini"}, "gpt-5.4-mini", None, None),
        ({"embedding_provider": "openai", "min_relevance": 0.3}, None, None, 0.3),
    ],
)
def test_provider_dependent_defaults(overrides, llm_model, embedding_model, gate):
    s = Settings(**overrides, _env_file=None)
    if llm_model is not None or overrides.get("llm_provider") == "none":
        assert s.llm_model == llm_model
    if embedding_model is not None:
        assert s.embedding_model == embedding_model
    if gate is not None:
        assert s.min_relevance == gate


def test_uncalibrated_embedding_model_disables_gate_and_flags_it():
    s = Settings(embedding_model="BAAI/bge-base-en-v1.5", _env_file=None)
    assert s.min_relevance == 0.0 and s.gate_uncalibrated
    assert not Settings(_env_file=None).gate_uncalibrated
    explicit = Settings(embedding_model="BAAI/bge-base-en-v1.5", min_relevance=0, _env_file=None)
    assert not explicit.gate_uncalibrated  # an explicit 0 is a deliberate choice


def test_openai_key_from_env_is_secret(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "openai")
    s = Settings(_env_file=None)
    assert s.openai_api_key.get_secret_value() == "sk-openai-secret"
    assert "sk-openai-secret" not in repr(s) and "sk-openai-secret" not in s.model_dump_json()
