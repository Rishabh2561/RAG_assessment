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
