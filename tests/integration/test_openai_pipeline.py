"""The full RAG pipeline with OpenAI providers (mocked SDK clients): swapping vendors
must not change grounding, citation or abstention behaviour."""

import json
import re
from types import SimpleNamespace

import pytest

from rag_generator.config import Settings
from rag_generator.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag_generator.errors import RAGError
from rag_generator.generation.anthropic_provider import AnthropicProvider
from rag_generator.generation.openai_provider import OpenAIProvider
from rag_generator.orchestration import RAGApplication
from rag_generator.orchestration.factory import build_embedder, build_llm
from tests.conftest import SAMPLE_DIR


def _chat_client(answer_for):
    def create(**kwargs):
        user = kwargs["messages"][1]["content"]
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps(answer_for(user)), refusal=None),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=40),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _cite_first(user):
    text = re.search(r'<source id="S1"[^>]*>\n(.*?)\n</source>', user, re.S).group(1)[:60]
    return {"answerable": True, "answer": f"{text} [S1]", "citations": ["S1"], "reason": ""}


def test_factory_selects_openai_providers():
    s = Settings(
        llm_provider="openai",
        embedding_provider="openai",
        openai_api_key="k",
        openai_base_url="http://localhost:11434/v1",
        _env_file=None,
    )
    llm = build_llm(s)
    assert isinstance(llm, OpenAIProvider) and llm.model_id == "gpt-5.5"
    assert llm._client_kwargs["base_url"] == "http://localhost:11434/v1"
    embedder = build_embedder(s)
    assert isinstance(embedder, OpenAIEmbeddingProvider)
    assert embedder.model_id == "openai:text-embedding-3-small"
    anthropic = Settings(llm_provider="anthropic", _env_file=None)
    assert isinstance(build_llm(anthropic), AnthropicProvider)


def test_grounded_answer_through_openai_provider(make_app):
    make_app().ingestion().ingest_paths([SAMPLE_DIR], "n")
    llm = OpenAIProvider("gpt-5.5", client=_chat_client(_cite_first))
    answer = make_app(llm=llm).query("n").ask("What is the minimum password length?")
    assert answer.grounding_status == "grounded"
    assert answer.citations[0].chunk_id == answer.passages[0].chunk.chunk_id
    assert answer.trace.llm_model == "gpt-5.5"
    assert (answer.trace.input_tokens, answer.trace.output_tokens) == (300, 40)


def test_abstention_through_openai_provider(make_app):
    make_app().ingestion().ingest_paths([SAMPLE_DIR], "n")
    llm = OpenAIProvider(
        "gpt-5.5",
        client=_chat_client(
            lambda user: {"answerable": False, "answer": "", "citations": [], "reason": "absent"}
        ),
    )
    answer = make_app(llm=llm).query("n").ask("Who is the CEO?")
    assert answer.grounding_status == "abstained" and answer.reason == "absent"


def test_openai_embeddings_power_ingestion_and_retrieval(tmp_path):
    vocabulary = ["password", "leave", "robot", "compost"]

    def embed(**kwargs):  # tiny deterministic "semantic" embedding: keyword presence
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=i, embedding=[float(w in t.lower()) + 0.01 for w in vocabulary]
                )
                for i, t in enumerate(kwargs["input"])
            ]
        )

    client = SimpleNamespace(embeddings=SimpleNamespace(create=embed))
    settings = Settings(
        data_dir=tmp_path, embedding_provider="openai", llm_provider="none", _env_file=None
    )
    app = RAGApplication(settings, embedder=OpenAIEmbeddingProvider(client=client))
    report = app.ingestion().ingest_paths([SAMPLE_DIR], "n")
    assert report.failed == 0
    assert app.repository.open("n").store.model_id == "openai:text-embedding-3-small"
    answer = app.query("n").ask("password rules")
    assert answer.passages[0].chunk.source == "it_security_policy.txt"
    assert settings.min_relevance == 0.0  # gate off until calibrated for this model


def test_missing_openai_package_gives_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai" or name.startswith("rag_generator.generation.openai_provider"):
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RAGError, match=r"\[openai\]"):
        build_llm(Settings(llm_provider="openai", _env_file=None))
