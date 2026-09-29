"""Command-line interface. A thin adapter: parse arguments, call services, print results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from rag_generator.config import Settings
from rag_generator.errors import RAGError
from rag_generator.models import Answer, IngestReport
from rag_generator.observability import configure_logging
from rag_generator.orchestration import RAGApplication

app = typer.Typer(
    help="RAG Generator: ingest any document set and ask grounded, cited questions.",
    no_args_is_help=True,
    add_completion=False,
)

CollectionOpt = Annotated[
    str | None, typer.Option("--collection", "-c", help="Collection name (default from config).")
]


def _settings(**overrides) -> Settings:
    try:
        settings = Settings(**{k: v for k, v in overrides.items() if v is not None})
    except ValidationError as exc:
        typer.secho(f"Configuration error:\n{exc}", fg="red", err=True)
        raise typer.Exit(2) from exc
    configure_logging(settings.log_level, settings.log_format, settings.log_content)
    return settings


def _fail(exc: Exception) -> typer.Exit:
    typer.secho(f"Error: {exc}", fg="red", err=True)
    return typer.Exit(1)


@app.command()
def ingest(
    paths: Annotated[list[Path], typer.Argument(exists=True, help="Files and/or directories.")],
    collection: CollectionOpt = None,
) -> None:
    """Parse, chunk, embed and index documents into a collection."""
    settings = _settings()
    name = collection or settings.default_collection
    try:
        report = RAGApplication(settings).ingestion().ingest_paths(paths, name)
    except RAGError as exc:
        raise _fail(exc) from exc
    _print_ingest(report)
    if report.results and report.succeeded == 0 and report.failed:
        raise typer.Exit(1)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    collection: CollectionOpt = None,
    mode: Annotated[str | None, typer.Option(help="dense | bm25 | hybrid")] = None,
    top_k: Annotated[int | None, typer.Option("--top-k", "-k", min=1, max=50)] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Retrieval only.")] = False,
    rewrite: Annotated[
        bool | None, typer.Option("--rewrite/--no-rewrite", help="Corrective query rewrite.")
    ] = None,
    show_trace: Annotated[bool, typer.Option("--trace", help="Print timing/trace.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full JSON result.")] = False,
) -> None:
    """Ask a question against a collection."""
    settings = _settings(
        retrieval_mode=mode,
        query_rewrite=rewrite,
        llm_provider="none" if no_llm else None,
    )
    name = collection or settings.default_collection
    try:
        answer = RAGApplication(settings).query(name, top_k=top_k).ask(question)
    except RAGError as exc:
        raise _fail(exc) from exc
    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
        return
    _print_answer(answer, show_trace)


@app.command()
def docs(collection: CollectionOpt = None) -> None:
    """List documents in a collection."""
    settings = _settings(llm_provider="none")
    name = collection or settings.default_collection
    try:
        records = RAGApplication(settings).repository.open(name).catalog.list()
    except RAGError as exc:
        raise _fail(exc) from exc
    if not records:
        typer.echo(f"Collection '{name}' is empty.")
    for r in records:
        pages = f"{r.num_pages} pages, " if r.num_pages else ""
        typer.echo(f"{r.doc_id}  {r.source}  ({r.file_type}, {pages}{r.num_chunks} chunks)")


@app.command()
def collections() -> None:
    """List collections."""
    settings = _settings(llm_provider="none")
    names = RAGApplication(settings).repository.list_names()
    typer.echo("\n".join(names) if names else "No collections yet.")


@app.command()
def delete(
    doc_id: Annotated[str, typer.Argument(help="Document id (see `rag docs`).")],
    collection: CollectionOpt = None,
) -> None:
    """Delete one document from a collection."""
    settings = _settings(llm_provider="none")
    name = collection or settings.default_collection
    try:
        record = RAGApplication(settings).ingestion().delete_document(name, doc_id)
    except RAGError as exc:
        raise _fail(exc) from exc
    if record is None:
        raise _fail(RAGError(f"document '{doc_id}' not found in '{name}'"))
    typer.echo(f"Deleted {record.source} ({doc_id}).")


@app.command()
def drop(
    collection: CollectionOpt = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete an entire collection and its index."""
    settings = _settings(llm_provider="none")
    name = collection or settings.default_collection
    if not yes:
        typer.confirm(f"Delete collection '{name}' and all its documents?", abort=True)
    try:
        RAGApplication(settings).repository.drop(name)
    except RAGError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Dropped collection '{name}'.")


@app.command("eval")
def evaluate(
    dataset: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="JSONL dataset.")],
    collection: CollectionOpt = None,
    modes: Annotated[
        str, typer.Option(help="Comma-separated retrieval modes.")
    ] = "dense,bm25,hybrid",
    rerankers: Annotated[str, typer.Option(help="Comma-separated: none,cross_encoder")] = "none",
    answers: Annotated[
        bool, typer.Option("--answers", help="Also evaluate generation (LLM).")
    ] = False,
    judge: Annotated[bool, typer.Option("--judge", help="LLM-judge faithfulness.")] = False,
    out_dir: Annotated[Path, typer.Option(help="Where to write the JSON report.")] = Path(
        "eval_reports"
    ),
) -> None:
    """Evaluate retrieval (and optionally answers) against a labelled dataset."""
    from rag_generator.evaluation import EvaluationRunner, load_dataset
    from rag_generator.orchestration.factory import build_reranker

    settings = _settings(llm_provider=None if answers else "none")
    name = collection or settings.default_collection
    items = load_dataset(dataset)
    application = RAGApplication(settings)
    try:
        runner = EvaluationRunner(application, name)
        report: dict = {"dataset": str(dataset), "collection": name, "n_items": len(items)}
        report["retrieval"] = []
        for reranker_name in [r.strip() for r in rerankers.split(",") if r.strip()]:
            reranker = build_reranker(settings.model_copy(update={"reranker": reranker_name}))
            for mode in [m.strip() for m in modes.split(",") if m.strip()]:
                result = runner.evaluate_retrieval(items, mode, reranker)
                report["retrieval"].append(result.__dict__)
        if answers:
            report["answers"] = runner.evaluate_answers(items, judge=judge)
    except RAGError as exc:
        raise _fail(exc) from exc

    _print_eval(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    typer.echo(f"\nFull report: {out_path}")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Run the REST API (Swagger UI at /docs)."""
    import uvicorn

    from rag_generator.interfaces.api import create_app

    settings = _settings()
    uvicorn.run(create_app(RAGApplication(settings)), host=host, port=port)


# --- Output formatting ----------------------------------------------------------------


def _print_ingest(report: IngestReport) -> None:
    colours = {
        "ingested": "green",
        "replaced": "cyan",
        "skipped_duplicate": "yellow",
        "failed": "red",
    }
    for r in report.results:
        detail = (
            f"{r.num_chunks} chunks" if r.status in ("ingested", "replaced") else (r.message or "")
        )
        typer.secho(f"  [{r.status}] {r.source}: {detail}", fg=colours[r.status])
        for warning in r.warnings:
            typer.secho(f"      warning: {warning}", fg="yellow")
    typer.echo(
        f"Collection '{report.collection}': {report.succeeded} ingested, {report.failed} failed "
        f"({report.duration_ms / 1000:.1f}s)"
    )


def _print_answer(answer: Answer, show_trace: bool) -> None:
    status_colour = {"grounded": "green", "unverified": "yellow", "abstained": "yellow"}
    if answer.answer is not None:
        typer.echo(f"\n{answer.answer}\n")
    typer.secho(f"status: {answer.grounding_status}", fg=status_colour.get(answer.grounding_status))
    if answer.reason:
        typer.echo(f"reason: {answer.reason}")
    if answer.citations:
        typer.echo("\nSources:")
        for c in answer.citations:
            page = f", p.{c.page}" if c.page else ""
            typer.echo(f'  [{c.source_id}] {c.source}{page} — "{c.snippet[:160]}"')
    elif answer.grounding_status == "retrieval_only":
        typer.echo("\nTop passages (retrieval-only mode):")
        for i, p in enumerate(answer.passages, start=1):
            page = f", p.{p.chunk.page}" if p.chunk.page else ""
            snippet = " ".join(p.chunk.text.split())[:200]
            typer.echo(f"  {i}. {p.chunk.source}{page} (score {p.score:.3f})\n     {snippet}")
    if show_trace:
        t = answer.trace
        typer.echo(
            f"\ntrace: mode={t.retrieval_mode} reranker={t.reranker} model={t.llm_model} "
            f"attempts={t.attempts} best_dense={t.best_dense_score} "
            f"tokens={t.input_tokens}/{t.output_tokens} total={t.total_ms:.0f}ms"
        )
        typer.echo("  " + ", ".join(f"{s.stage}={s.duration_ms:.0f}ms" for s in t.timings))
        if t.rewritten_queries:
            typer.echo(f"  rewrites: {t.rewritten_queries}")


def _print_eval(report: dict) -> None:
    typer.echo(f"\nRetrieval ({report['n_items']} items, collection '{report['collection']}')")
    columns = ("hit@1", "hit@3", "hit@5", "mrr")
    header = f"{'mode':<8} {'reranker':<14} " + " ".join(f"{c:>6}" for c in columns) + "   p50ms"
    typer.echo(header)
    for r in report["retrieval"]:
        m = r["metrics"]
        typer.echo(
            f"{r['mode']:<8} {r['reranker']:<14} {m['hit@1']:>6.3f} {m['hit@3']:>6.3f} "
            f"{m['hit@5']:>6.3f} {m['mrr@10']:>6.3f} {m['latency_p50_ms']:>7.1f}"
        )
    for r in report["retrieval"]:
        if r["calibration"]:
            c = r["calibration"]
            typer.echo(
                f"\nGate calibration ({r['mode']}): suggested RAG_MIN_RELEVANCE="
                f"{c['suggested_min_relevance']} (accuracy {c['gate_accuracy_at_suggested']:.2f}; "
                f"answerable min {c['answerable_best_dense_min']:.3f}, "
                f"unanswerable max {c['unanswerable_best_dense_max']:.3f})"
            )
            break
    if "answers" in report:
        typer.echo("\nAnswers")
        for key, value in report["answers"]["metrics"].items():
            typer.echo(
                f"  {key:<24} {value:.3f}" if isinstance(value, float) else f"  {key:<24} {value}"
            )
