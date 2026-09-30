"""Sidebar section where the user picks the answer model: the server's, or their own
Anthropic / OpenAI key. The key lives only in this browser session and is sent to the
API as request headers; the server never stores it."""

from __future__ import annotations

from typing import Any

import streamlit as st

from rag_generator.ui.client import APIError, RAGClient
from rag_generator.ui.formatting import PROVIDER_LABELS, detect_provider, mask_key

SERVER_DEFAULT = "Server default"
OWN_KEY = "My API key"
AUTO = "Auto-detect"


def render_llm_settings(client: RAGClient, config: dict[str, Any]) -> dict[str, Any]:
    """Render the section and return ``config`` with ``llm_provider`` / ``llm_model``
    set to what answers will actually use, so the views enable the right controls."""
    st.sidebar.subheader("Answer model")
    if not config.get("accepts_client_llm_keys", False):
        client.set_llm(None)
        st.sidebar.caption("This server uses its own LLM; entering a key is disabled.")
        return config

    source = st.sidebar.radio(
        "Answer with",
        [SERVER_DEFAULT, OWN_KEY],
        horizontal=True,
        key="llm_source",
        index=0 if config["llm_provider"] != "none" else 1,
    )
    if source == SERVER_DEFAULT:
        client.set_llm(None)
        _describe_server_llm(config)
        return config

    api_key = st.sidebar.text_input(
        "API key",
        type="password",
        key="llm_api_key",
        placeholder="sk-ant-… or sk-…",
        help="Anthropic (sk-ant-…) or OpenAI (sk-…). Kept only in this browser session and "
        "sent to the API with each request; never saved or logged.",
    ).strip()
    choice = st.sidebar.selectbox(
        "Provider", [AUTO, *PROVIDER_LABELS.values()], key="llm_provider_choice"
    )
    provider = _resolve_provider(api_key, choice)
    default_model = config.get("default_llm_models", {}).get(provider or "", "")
    model = st.sidebar.text_input(
        "Model (optional)",
        key="llm_model_override",
        placeholder=default_model or "provider default",
    ).strip()

    if not api_key:
        client.set_llm(None)
        st.sidebar.info("Enter an Anthropic or OpenAI API key to get generated answers.")
        return {**config, "llm_provider": "none", "llm_model": None}
    if provider is None:
        client.set_llm(None)
        st.sidebar.warning("Couldn't tell which provider this key is for. Choose it above.")
        return {**config, "llm_provider": "none", "llm_model": None}

    client.set_llm(api_key, provider, model or None)
    label = PROVIDER_LABELS[provider]
    st.sidebar.caption(
        f"Using **{label}** · `{model or default_model}` · key `{mask_key(api_key)}`"
    )
    _check_key_button(client, api_key, provider, model)
    return {**config, "llm_provider": provider, "llm_model": model or default_model}


def _resolve_provider(api_key: str, choice: str) -> str | None:
    if choice != AUTO:
        return next(k for k, v in PROVIDER_LABELS.items() if v == choice)
    return detect_provider(api_key) if api_key else None


def _describe_server_llm(config: dict[str, Any]) -> None:
    if config["llm_provider"] == "none":
        st.sidebar.info(
            "The server has no LLM key, so answers are retrieval-only. "
            "Choose “My API key” to use your own Anthropic or OpenAI key."
        )
    else:
        label = PROVIDER_LABELS.get(config["llm_provider"], config["llm_provider"])
        auto = " (chosen from the server's keys)" if config.get("llm_provider_auto") else ""
        st.sidebar.caption(f"Using **{label}** · `{config['llm_model']}`{auto}")


def _check_key_button(client: RAGClient, api_key: str, provider: str, model: str) -> None:
    # Remember the result per (key, provider, model) so it resets when any of them changes.
    fingerprint = (mask_key(api_key), len(api_key), provider, model)
    results = st.session_state.setdefault("llm_checks", {})
    if st.sidebar.button("Check key", key="llm_check", help="Makes one tiny LLM request."):
        try:
            results[fingerprint] = client.verify_llm()
        except APIError as exc:
            results[fingerprint] = {"ok": False, "detail": str(exc)}
    result = results.get(fingerprint)
    if result is None:
        return
    if result.get("ok"):
        st.sidebar.success(f"Key works · {result.get('model')}")
    else:
        st.sidebar.error(f"Key check failed: {explain_key_error(result.get('detail'), provider)}")


def explain_key_error(detail: str | None, provider: str) -> str:
    """Rephrase server-operator advice ("set ANTHROPIC_API_KEY") for a key typed here."""
    detail = detail or "unknown error"
    if "authentication failed" in detail:
        return f"{PROVIDER_LABELS[provider]} rejected this API key. Check it and try again."
    return detail
