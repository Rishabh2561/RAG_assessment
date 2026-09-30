"""Streamlit entry point: `rag ui` (or `streamlit run src/rag_generator/ui/app.py`).

The sidebar connects to the REST API and picks a collection; the tabs render the
views. Configure the API location with RAG_UI_API_URL (default http://127.0.0.1:8000).
"""

from __future__ import annotations

import os
import re

import streamlit as st

from rag_generator.ui.client import APIError, RAGClient
from rag_generator.ui.llm_settings import render_llm_settings
from rag_generator.ui.views import render_ask, render_documents, render_evaluate, render_system

DEFAULT_API_URL = "http://127.0.0.1:8000"
NEW_COLLECTION = "+ New collection…"
_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # mirrors the server's rule


def get_client(api_url: str) -> RAGClient:
    """One client per API URL for the session. Tests inject ``session_state['client']``."""
    if "client" in st.session_state:
        return st.session_state["client"]
    clients = st.session_state.setdefault("_clients", {})
    if api_url not in clients:
        clients[api_url] = RAGClient(api_url)
    return clients[api_url]


def sidebar() -> tuple[RAGClient, dict, dict, list[str]]:
    st.sidebar.title("📚 RAG Generator")
    api_url = st.sidebar.text_input(
        "API URL", value=os.environ.get("RAG_UI_API_URL", DEFAULT_API_URL), key="api_url"
    )
    client = get_client(api_url)
    try:
        health, config, collections = client.health(), client.config(), client.collections()
    except APIError as exc:
        st.sidebar.error(str(exc))
        st.error("The UI needs the REST API. Start it in another terminal with `rag serve`.")
        st.stop()
    st.sidebar.success("Connected")
    st.sidebar.caption(f"Embeddings: {config['embedding_model']}")
    return client, health, config, collections


def choose_collection(collections: list[str]) -> tuple[str | None, bool]:
    """Returns (collection name, whether it exists on the server)."""
    remembered = st.session_state.get("collection")
    options = [*collections, NEW_COLLECTION]
    index = options.index(remembered) if remembered in collections else 0
    choice = st.sidebar.selectbox("Collection", options, index=index, key="collection_select")
    if choice != NEW_COLLECTION:
        st.session_state["collection"] = choice
        return choice, True
    name = st.sidebar.text_input(
        "New collection name",
        value=remembered if remembered and remembered not in collections else "",
        placeholder="e.g. hr-policies",
        key="new_collection",
    ).strip()
    if not name:
        st.sidebar.caption("Name it, then upload documents in the Documents tab.")
        return None, False
    if not _VALID_NAME.match(name):
        st.sidebar.error("Use 1-64 letters, digits, '_' or '-'.")
        return None, False
    st.session_state["collection"] = name
    return name, False


def main() -> None:
    st.set_page_config(page_title="RAG Generator", page_icon="📚", layout="wide")
    client, health, config, collections = sidebar()
    collection, exists = choose_collection(collections)
    st.sidebar.divider()
    config = render_llm_settings(client, config)  # the LLM answers will actually use
    st.sidebar.divider()
    st.sidebar.caption(
        "Each collection is an isolated document set. Answers cite only its documents."
    )

    ask, documents, evaluate, system = st.tabs(
        ["💬 Ask", "📄 Documents", "📊 Evaluate", "⚙️ System"]
    )
    with system:
        render_system(health, config)
    if collection is None:
        for tab in (ask, documents, evaluate):
            tab.info("Choose or create a collection in the sidebar.")
        return
    with ask:
        render_ask(client, collection, config, exists)
    with documents:
        render_documents(client, collection, config, exists)
    with evaluate:
        render_evaluate(client, collection, config, exists)


main()
