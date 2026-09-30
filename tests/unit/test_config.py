import pytest
from pydantic import ValidationError

from rag_generator.config import Settings


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    """Keep these tests independent of keys set on the developer's machine."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RAG_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_are_valid():
    s = Settings(_env_file=None)
    assert s.chunk_overlap < s.chunk_size
    assert s.top_k <= s.candidate_pool
    assert s.llm_provider == "none" and s.llm_provider_auto  # auto: no key -> retrieval only


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
        ({"llm_provider": "anthropic"}, "claude-opus-5-5", "BAAI/bge-small-en-v1.5", 0.50),
        ({"llm_provider": "openai"}, "gpt-5.5", "BAAI/bge-small-en-v1.5", 0.50),
        ({"embedding_provider": "openai"}, None, "text-embedding-3-small", 0.0),
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


@pytest.mark.parametrize(
    "keys,provider,model",
    [
        ({}, "none", None),
        ({"ANTHROPIC_API_KEY": "sk-ant-x"}, "anthropic", "claude-opus-5-5"),
        ({"OPENAI_API_KEY": "sk-proj-x"}, "openai", "gpt-5.5"),
        ({"ANTHROPIC_API_KEY": "sk-ant-x", "OPENAI_API_KEY": "sk-proj-x"}, "anthropic", None),
        ({"ANTHROPIC_API_KEY": "  "}, "none", None),  # blank key counts as absent
    ],
)
def test_auto_provider_follows_whichever_key_is_configured(monkeypatch, keys, provider, model):
    for var, value in keys.items():
        monkeypatch.setenv(var, value)
    s = Settings(_env_file=None)
    assert s.llm_provider == provider
    if model:
        assert s.llm_model == model


def test_explicit_provider_is_not_auto(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-x")
    s = Settings(llm_provider="anthropic", _env_file=None)
    assert s.llm_provider == "anthropic" and not s.llm_provider_auto


@pytest.mark.parametrize(
    "key,expected",
    [
        ("sk-ant-api03-abc", "anthropic"),
        ("  sk-ant-admin01-abc ", "anthropic"),
        ("sk-proj-abc", "openai"),
        ("sk-svcacct-abc", "openai"),
        ("sk-abc123", "openai"),
        ("AIzaSy-not-supported", None),
        ("", None),
    ],
)
def test_detect_provider_from_key_format(key, expected):
    from rag_generator.config import detect_provider

    assert detect_provider(key) == expected


def test_blank_or_comment_key_values_count_as_unset(tmp_path):
    """Regression: dotenv parses `KEY=   # note` as the value '# note'."""
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=   # set me\nOPENAI_API_KEY=\n")
    s = Settings(_env_file=env)
    assert s.anthropic_api_key is None and s.openai_api_key is None
    assert s.llm_provider == "none"


def test_env_example_resolves_to_retrieval_only_until_a_key_is_added():
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / ".env.example"
    s = Settings(_env_file=example)
    assert s.anthropic_api_key is None and s.openai_api_key is None
    assert s.llm_provider == "none" and s.allow_client_llm_keys
