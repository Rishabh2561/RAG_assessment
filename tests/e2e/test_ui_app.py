"""The Streamlit UI end to end: AppTest drives the real app script, whose RAGClient talks
HTTP to the real FastAPI app (in-process TestClient) backed by the real services with
the hashing embedder and a scripted LLM. No servers or network needed.
"""

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import rag_generator.ui
from rag_generator.interfaces.api import create_app
from rag_generator.ui.client import RAGClient
from tests.conftest import ALT_SAMPLE_DIR, SAMPLE_DIR, FakeLLM, cite_first_source

APP = str(Path(rag_generator.ui.__file__).parent / "app.py")


@pytest.fixture
def backend(make_app):
    """(RAGClient over HTTP to an in-process API, the scripted LLM)."""
    llm = FakeLLM(cite_first_source)
    api = TestClient(create_app(make_app(llm=llm)))
    return RAGClient("http://testserver", http_client=api), llm


def _ingest(client: RAGClient, directory: Path, name: str) -> None:
    client.upload(name, [(p.name, p.read_bytes()) for p in directory.iterdir()])


def _app(client: RAGClient) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["client"] = client
    return at.run()


def _texts(at: AppTest) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.caption]
    parts += (
        [e.value for e in at.success] + [e.value for e in at.info] + [e.value for e in at.error]
    )
    return "\n".join(str(p) for p in parts)


def test_starts_connected_and_lists_collections(backend):
    client, _ = backend
    _ingest(client, SAMPLE_DIR, "northwind")
    _ingest(client, ALT_SAMPLE_DIR, "garden")
    at = _app(client)
    assert not at.exception
    assert "Connected" in at.sidebar.success[0].value
    assert at.sidebar.selectbox(key="collection_select").options == [
        "garden",
        "northwind",
        "+ New collection…",
    ]


def test_ask_shows_grounded_answer_with_citations_and_trace(backend):
    client, llm = backend
    _ingest(client, SAMPLE_DIR, "northwind")
    at = _app(client)
    at.sidebar.selectbox(key="collection_select").set_value("northwind").run()
    at.toggle(key="opt_retrieval_only_True").set_value(False)
    at.chat_input[0].set_value("What is the minimum password length?").run()
    assert not at.exception
    text = _texts(at)
    assert "Grounded" in text or any("Grounded" in str(b.proto) for b in at.get("badge"))
    assert ":blue-background[S1]" in text  # inline citation chip and source card
    assert "**Sources**" in text
    assert len(llm.calls) == 1
    # The turn is kept in the chat history across reruns.
    at.run()
    assert "What is the minimum password length?" in _texts(at)


def test_retrieval_only_mode_lists_passages(backend):
    client, llm = backend
    _ingest(client, SAMPLE_DIR, "northwind")
    at = _app(client)
    at.sidebar.selectbox(key="collection_select").set_value("northwind").run()
    at.toggle(key="opt_retrieval_only_True").set_value(True)
    at.chat_input[0].set_value("E-221 battery overheating").run()
    assert not at.exception
    assert "nw200\\_operator\\_manual\\.pdf" in _texts(at)  # escaped source label on a card
    assert llm.calls == []


def test_documents_tab_lists_and_deletes(backend):
    client, _ = backend
    _ingest(client, ALT_SAMPLE_DIR, "garden")
    at = _app(client)
    assert at.dataframe  # document table rendered
    select = at.multiselect(key="delete_select")
    target = next(o for o in select.options if o.startswith("composting_guide.txt"))
    select.set_value([target]).run()  # the button enables on the rerun after selection
    at.button(key="delete_docs").click().run()
    assert not at.exception
    assert [d["source"] for d in client.documents("garden")] == ["community_garden_rules.md"]


def test_drop_collection_requires_confirmation(backend):
    client, _ = backend
    _ingest(client, ALT_SAMPLE_DIR, "garden")
    at = _app(client)
    assert at.button(key="drop").disabled
    at.checkbox(key="confirm_drop").check().run()
    at.button(key="drop").click().run()
    assert not at.exception
    assert client.collections() == []


def test_new_collection_flow_validates_name(backend):
    client, _ = backend
    at = _app(client)  # no collections yet -> "+ New collection…" is the only option
    at.sidebar.text_input(key="new_collection").input("bad name!").run()
    assert "Use 1-64 letters" in at.sidebar.error[0].value
    at.sidebar.text_input(key="new_collection").input("hr-policies").run()
    assert not at.exception
    assert "has no documents yet" in _texts(at)


def test_saved_evaluation_report_renders(backend):
    client, _ = backend
    _ingest(client, SAMPLE_DIR, "northwind")
    dataset = (Path(__file__).parents[2] / "data" / "eval" / "northwind_eval.jsonl").read_bytes()
    report = client.evaluate(
        "northwind", ("set.jsonl", dataset), modes=["dense", "bm25"], rerankers=["none"]
    )
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["client"] = client
    at.session_state["collection"] = "northwind"
    at.session_state["eval_reports"] = {"northwind": report}
    at.run()
    assert not at.exception
    assert "Best configuration" in _texts(at)
    assert at.get("download_button")


def test_unreachable_api_shows_guidance():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    at = _app(RAGClient("http://nowhere.test", transport=httpx.MockTransport(refuse)))
    assert not at.exception
    assert "rag serve" in _texts(at) + "\n".join(e.value for e in at.sidebar.error)


# --- Entering an Anthropic / OpenAI key in the sidebar --------------------------------


@pytest.fixture
def keyless_backend(make_app, monkeypatch):
    """API server with no LLM of its own; client keys build recording fakes."""
    import rag_generator.orchestration.factory as factory

    built: list[dict] = []

    def fake_build_llm(settings, *, provider=None, api_key=None, model=None):
        if provider is None:
            return None
        built.append({"provider": provider, "api_key": api_key, "model": model})
        return FakeLLM(cite_first_source)

    monkeypatch.setattr(factory, "build_llm", fake_build_llm)
    api = TestClient(create_app(make_app()))
    client = RAGClient("http://testserver", http_client=api)
    _ingest(client, SAMPLE_DIR, "northwind")
    return client, built


def _keyless_app(client):
    at = _app(client)
    at.sidebar.selectbox(key="collection_select").set_value("northwind").run()
    return at


@pytest.mark.parametrize(
    "key,provider,label",
    [
        ("sk-ant-api03-secret-abcd", "anthropic", "Anthropic"),
        ("sk-proj-secret-wxyz", "openai", "OpenAI"),
    ],
)
def test_entered_key_selects_provider_and_answers(keyless_backend, key, provider, label):
    client, built = keyless_backend
    at = _keyless_app(client)
    assert at.sidebar.radio(key="llm_source").value == "My API key"  # server has no LLM
    at.sidebar.text_input(key="llm_api_key").input(key).run()
    sidebar_text = "\n".join(c.value for c in at.sidebar.caption)
    assert f"Using **{label}**" in sidebar_text
    assert key not in sidebar_text  # only a masked preview is shown
    at.toggle(key="opt_retrieval_only_True").set_value(False)
    at.chat_input[0].set_value("What is the minimum password length?").run()
    assert not at.exception
    assert ":blue-background[S1]" in _texts(at)  # generated, cited answer
    assert built == [{"provider": provider, "api_key": key, "model": None}]


def test_unrecognised_key_needs_explicit_provider(keyless_backend):
    client, built = keyless_backend
    at = _keyless_app(client)
    at.sidebar.text_input(key="llm_api_key").input("custom-gateway-key").run()
    assert "Couldn't tell which provider" in at.sidebar.warning[0].value
    at.sidebar.selectbox(key="llm_provider_choice").set_value("OpenAI").run()
    at.sidebar.text_input(key="llm_model_override").input("gpt-5.4-mini").run()
    at.toggle(key="opt_retrieval_only_True").set_value(False)
    at.chat_input[0].set_value("password length?").run()
    assert built[-1] == {
        "provider": "openai",
        "api_key": "custom-gateway-key",
        "model": "gpt-5.4-mini",
    }


def test_check_key_button(keyless_backend):
    client, _ = keyless_backend
    at = _keyless_app(client)
    at.sidebar.text_input(key="llm_api_key").input("sk-ant-api03-secret-abcd").run()
    at.sidebar.button(key="llm_check").click().run()
    assert "Key works" in at.sidebar.success[-1].value


def test_without_a_key_answers_stay_retrieval_only(keyless_backend):
    client, built = keyless_backend
    at = _keyless_app(client)
    assert "Enter an Anthropic or OpenAI API key" in at.sidebar.info[0].value
    assert at.toggle(key="opt_retrieval_only_False").disabled
    at.chat_input[0].set_value("password length?").run()
    assert built == [] and "Retrieval only" in _texts(at)


def test_server_can_refuse_client_keys(make_app):
    api = TestClient(create_app(make_app(allow_client_llm_keys=False)))
    at = _app(RAGClient("http://testserver", http_client=api))
    assert "entering a key is disabled" in "\n".join(c.value for c in at.sidebar.caption)
