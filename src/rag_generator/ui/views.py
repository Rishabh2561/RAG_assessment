"""The four views of the UI: Ask, Documents, Evaluate, System.

Each view receives the API client, the active collection and the server's ``/config``
so that nothing (file types, limits, modes, defaults) is hard-coded in the UI.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from rag_generator.ui.client import APIError, RAGClient
from rag_generator.ui.formatting import (
    answer_metric_rows,
    category_rows,
    document_rows,
    highlight_citations,
    ingest_rows,
    ingest_summary,
    md_escape,
    passage_rows,
    retrieval_metric_rows,
    source_label,
    status_badge,
    timing_series,
)

# --- Ask -------------------------------------------------------------------------------


def render_ask(client: RAGClient, collection: str, config: dict[str, Any], exists: bool) -> None:
    if not exists:
        st.info(f"Collection **{collection}** has no documents yet. Add some in the Documents tab.")
        return
    options = _query_options(config)
    history: list[dict] = st.session_state.setdefault("history", {}).setdefault(collection, [])

    header, clear = st.columns([5, 1])
    header.caption(
        "Answers come only from this collection's documents. "
        ":blue-background[S1] markers link claims to the cited passages."
    )
    if history and clear.button("Clear chat", key="clear_chat", width="stretch"):
        history.clear()
        st.rerun()

    for turn in history:
        _render_turn(turn)

    question = st.chat_input(
        f"Ask a question about '{collection}'…", max_chars=config["max_question_chars"]
    )
    if question:
        turn: dict[str, Any] = {"question": question}
        with st.spinner("Retrieving passages and generating an answer…"):
            try:
                turn["answer"] = client.query(collection, question, **options)
            except APIError as exc:
                turn["error"] = _error_text(exc)
        history.append(turn)
        _render_turn(turn)


def _query_options(config: dict[str, Any]) -> dict[str, Any]:
    defaults = config["defaults"]
    llm_available = config["llm_provider"] != "none"
    with st.expander("Query options", expanded=False):
        left, middle, right = st.columns(3)
        modes = config["retrieval_modes"]
        mode = left.selectbox(
            "Retrieval mode",
            modes,
            index=modes.index(defaults["retrieval_mode"]),
            help="dense = semantic; bm25 = keyword; hybrid = both, fused by rank.",
            key="opt_mode",
        )
        top_k = middle.slider("Passages given to the LLM", 1, 20, defaults["top_k"], key="opt_k")
        retrieval_only = right.toggle(
            "Retrieval only",
            value=not llm_available,
            disabled=not llm_available,
            help="Skip generation and show ranked passages."
            + ("" if llm_available else " (Enter an API key in the sidebar to get answers.)"),
            # Keyed on availability so the toggle resets when a key is entered or removed.
            key=f"opt_retrieval_only_{llm_available}",
        )
        rewrite = right.toggle(
            "Corrective rewrite",
            value=defaults["query_rewrite"],
            disabled=retrieval_only,
            help="If the context looks insufficient, rephrase the question and retry once.",
            key="opt_rewrite",
        )
    return {
        "mode": mode,
        "top_k": top_k,
        "retrieval_only": retrieval_only,
        "rewrite": rewrite,
        "include_passages": True,
    }


def _render_turn(turn: dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.markdown(md_escape(turn["question"]))
    with st.chat_message("assistant"):
        if "error" in turn:
            st.error(turn["error"])
        else:
            render_answer(turn["answer"])


def render_answer(answer: dict[str, Any]) -> None:
    label, colour, explanation = status_badge(answer["grounding_status"])
    st.badge(label, color=colour)
    if answer["grounding_status"] != "retrieval_only" and answer.get("answer"):
        st.markdown(highlight_citations(answer["answer"]))
    if answer.get("reason"):
        st.caption(f"Reason: {md_escape(answer['reason'])}")
    elif explanation:
        st.caption(explanation)

    if answer.get("citations"):
        st.markdown("**Sources**")
        for c in answer["citations"]:
            with st.container(border=True):
                label = md_escape(source_label(c["source"], c["page"]))
                st.markdown(f":blue-background[{c['source_id']}] **{label}**")
                st.caption(md_escape(c["snippet"]))

    passages = answer.get("passages") or []
    if answer["grounding_status"] == "retrieval_only":
        for i, p in enumerate(passages, start=1):
            chunk = p["chunk"]
            with st.container(border=True):
                st.markdown(
                    f"**{i}. {md_escape(source_label(chunk['source'], chunk.get('page')))}** "
                    f"· score {p['score']:.3f}"
                )
                st.caption(md_escape(" ".join(chunk["text"].split())[:600]))
    elif passages:
        with st.expander(f"Retrieved passages ({len(passages)})"):
            st.dataframe(passage_rows(passages), hide_index=True, width="stretch")
    _render_trace(answer["trace"])


def _render_trace(trace: dict[str, Any]) -> None:
    with st.expander("Trace: how this answer was produced"):
        cols = st.columns(4)
        cols[0].metric("Total", f"{trace['total_ms']:.0f} ms")
        cols[1].metric("LLM calls", trace["attempts"] + (1 if trace["rewritten_queries"] else 0))
        cols[2].metric("Tokens in / out", f"{trace['input_tokens']} / {trace['output_tokens']}")
        best = trace.get("best_dense_score")
        gate = trace.get("gate_threshold")
        cols[3].metric(
            "Best similarity",
            "n/a" if best is None else f"{best:.3f}",
            help=f"Relevance gate threshold: {gate if gate is not None else 'off'}",
        )
        timings = timing_series(trace)
        if timings:
            st.bar_chart(timings, horizontal=True, x_label="ms", y_label="stage")
        st.caption(
            f"mode **{trace['retrieval_mode']}** · reranker **{trace['reranker']}** · "
            f"model **{trace.get('llm_model') or 'none'}**"
        )
        if trace["rewritten_queries"]:
            st.markdown(
                "**Rewritten queries:** " + "; ".join(map(md_escape, trace["rewritten_queries"]))
            )
        if trace.get("rewrite_error"):
            st.warning(f"Rewrite skipped: {trace['rewrite_error']}")
        if trace["invalid_citations"]:
            st.warning(
                f"Dropped citations to sources that were not provided: {trace['invalid_citations']}"
            )
        st.code("\n".join(trace["retrieved_chunk_ids"]) or "(none)", language=None)


# --- Documents -------------------------------------------------------------------------


def render_documents(
    client: RAGClient, collection: str, config: dict[str, Any], exists: bool
) -> None:
    st.subheader(f"Add documents to '{collection}'")
    nonce = st.session_state.setdefault("upload_nonce", 0)
    files = st.file_uploader(
        "Files",
        type=config["supported_extensions"],
        accept_multiple_files=True,
        key=f"uploader_{nonce}",
        help="Re-uploading identical content is skipped; a changed file with the same name "
        "replaces the old version.",
    )
    extensions = ", ".join("." + e for e in config["supported_extensions"])
    st.caption(
        f"Supported: {extensions} · up to {config['max_file_mb']:g} MB per file, "
        f"{config['max_upload_files']} files per upload"
    )
    if st.button("Ingest", type="primary", disabled=not files, key="ingest"):
        if len(files) > config["max_upload_files"]:
            st.error(f"Select at most {config['max_upload_files']} files per upload.")
        else:
            with st.spinner(f"Parsing, chunking and embedding {len(files)} file(s)…"):
                try:
                    report = client.upload(collection, [(f.name, f.getvalue()) for f in files])
                except APIError as exc:
                    st.error(_error_text(exc))
                else:
                    st.session_state["last_ingest"] = report
                    st.session_state["upload_nonce"] = nonce + 1  # clears the uploader
                    st.session_state["collection"] = collection
                    st.rerun()

    report = st.session_state.get("last_ingest")
    if report and report.get("collection") == collection:
        ok, skipped, failed = ingest_summary(report)
        message = f"{ok} indexed · {skipped} skipped (duplicate) · {failed} failed"
        (st.warning if failed else st.success)(message)
        st.dataframe(ingest_rows(report), hide_index=True, width="stretch")

    st.divider()
    _render_document_list(client, collection, exists)


def _render_document_list(client: RAGClient, collection: str, exists: bool) -> None:
    st.subheader("Documents in this collection")
    if not exists:
        st.caption("The collection is created when the first document is ingested.")
        return
    try:
        documents = client.documents(collection)
    except APIError as exc:
        st.error(_error_text(exc))
        return
    if not documents:
        st.caption("No documents.")
        return
    rows = document_rows(documents)
    total_chunks = sum(d["num_chunks"] for d in documents)
    st.caption(f"{len(documents)} document(s), {total_chunks} chunks")
    st.dataframe(rows, hide_index=True, width="stretch")

    by_label = {f"{d['source']} ({d['doc_id']})": d["doc_id"] for d in documents}
    left, right = st.columns([4, 1])
    selected = left.multiselect("Delete documents", list(by_label), key="delete_select")
    if right.button("Delete", disabled=not selected, key="delete_docs", width="stretch"):
        for label in selected:
            try:
                client.delete_document(collection, by_label[label])
            except APIError as exc:
                st.error(_error_text(exc))
        st.rerun()

    with st.expander("Danger zone"):
        confirm = st.checkbox(
            f"I want to delete the whole collection '{collection}'", key="confirm_drop"
        )
        if st.button("Drop collection", disabled=not confirm, type="primary", key="drop"):
            try:
                client.drop_collection(collection)
            except APIError as exc:
                st.error(_error_text(exc))
            else:
                st.session_state.pop("collection", None)
                st.session_state.get("history", {}).pop(collection, None)
                st.rerun()


# --- Evaluate --------------------------------------------------------------------------


def render_evaluate(
    client: RAGClient, collection: str, config: dict[str, Any], exists: bool
) -> None:
    st.subheader("Evaluate retrieval and answers")
    st.caption(
        "Upload a JSONL dataset of labelled questions (format: docs/06_EVALUATION.md, "
        "example: data/eval/northwind_eval.jsonl). Retrieval metrics need no LLM."
    )
    if not exists:
        st.info("Ingest documents into this collection first.")
        return
    llm_available = config["llm_provider"] != "none"
    dataset = st.file_uploader("Dataset (.jsonl)", type=["jsonl"], key="eval_dataset")
    left, right = st.columns(2)
    modes = left.multiselect(
        "Retrieval modes",
        config["retrieval_modes"],
        default=config["retrieval_modes"],
        key="eval_modes",
    )
    rerankers = right.multiselect(
        "Rerankers",
        config["rerankers"],
        default=["none"],
        key="eval_rerankers",
        help="cross_encoder downloads a local model on first use and adds latency.",
    )
    answers = left.toggle(
        "Also evaluate answers (calls the LLM for every question)",
        disabled=not llm_available,
        key="eval_answers",
    )
    judge = right.toggle("LLM-judge faithfulness", disabled=not answers, key="eval_judge")

    if st.button(
        "Run evaluation", type="primary", disabled=not (dataset and modes), key="run_eval"
    ):
        with st.spinner("Evaluating… (answer evaluation can take a few minutes)"):
            try:
                report = client.evaluate(
                    collection,
                    (dataset.name, dataset.getvalue()),
                    modes=modes,
                    rerankers=rerankers or ["none"],
                    answers=answers,
                    judge=judge,
                )
            except APIError as exc:
                st.error(_error_text(exc))
            else:
                st.session_state.setdefault("eval_reports", {})[collection] = report

    report = st.session_state.get("eval_reports", {}).get(collection)
    if report:
        render_eval_report(report)


def render_eval_report(report: dict[str, Any]) -> None:
    st.markdown(f"**Retrieval** · {report['n_items']} questions · dataset `{report['dataset']}`")
    rows = retrieval_metric_rows(report)
    st.dataframe(rows, hide_index=True, width="stretch")
    scored = [r for r in rows if isinstance(r["MRR@10"], float)]
    if scored:
        best = max(scored, key=lambda r: (r["hit@5"], r["MRR@10"]))
        st.success(
            f"Best configuration: **{best['mode']} / {best['reranker']}** "
            f"(hit@5 {best['hit@5']}, MRR {best['MRR@10']})"
        )
    calibration = next((r["calibration"] for r in report["retrieval"] if r["calibration"]), None)
    if calibration:
        st.info(
            "Relevance gate: suggested "
            f"**RAG_MIN_RELEVANCE={calibration['suggested_min_relevance']}**. "
            f"Answerable questions scored ≥ {calibration['answerable_best_dense_min']:.3f}; "
            f"unanswerable ones up to {calibration['unanswerable_best_dense_max']:.3f}. "
            "Keep the threshold below the answerable minimum."
        )
    with st.expander("hit@5 by question category"):
        st.dataframe(category_rows(report), hide_index=True, width="stretch")
    if report.get("answers"):
        st.markdown("**Answers**")
        st.dataframe(answer_metric_rows(report), hide_index=True, width="stretch")
    st.download_button(
        "Download full report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"eval_{report['collection']}.json",
        mime="application/json",
        key="download_report",
    )


# --- System ----------------------------------------------------------------------------


def render_system(health: dict[str, Any], config: dict[str, Any]) -> None:
    st.subheader("Server configuration")
    cols = st.columns(3)
    cols[0].metric("LLM", config["llm_model"] or "none", help=f"provider: {config['llm_provider']}")
    cols[1].metric("Embeddings", config["embedding_model"])
    cols[2].metric("Version", health.get("version", "?"))
    defaults = config["defaults"]
    st.dataframe(
        [{"setting": k, "value": str(v)} for k, v in defaults.items()],
        hide_index=True,
        width="stretch",
    )
    if config["defaults"].get("min_relevance") == 0:
        st.warning("The relevance gate is off. Calibrate it with the Evaluate tab.")
    st.caption("Server settings are changed with RAG_* environment variables (see .env.example).")
    with st.expander("Raw /config and /health"):
        st.json({"config": config, "health": health})


def _error_text(exc: APIError) -> str:
    prefix = f"{exc.error}: " if exc.error else ""
    hint = " This is temporary; try again shortly." if exc.retryable else ""
    if "authentication failed" in str(exc):
        return (
            f"{prefix}The provider rejected the API key. If you entered one in the sidebar, "
            "check it (use “Check key”); otherwise the server's key is invalid."
        )
    elif "credentials" in str(exc):
        hint = (
            " In the UI you can switch on **Retrieval only** under Query options, or set the "
            "API key on the server and restart `rag serve`."
        )
    return f"{prefix}{exc}{hint}"
