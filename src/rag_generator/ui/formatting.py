"""Pure presentation helpers for the UI (no Streamlit imports, unit-tested)."""

from __future__ import annotations

import re
from typing import Any

# grounding_status -> (label, badge colour, explanation shown to the user)
STATUS = {
    "grounded": ("Grounded", "green", "Every claim is tied to a cited passage."),
    "unverified": (
        "Unverified",
        "orange",
        "The model answered but cited no passage it was given. Treat with caution.",
    ),
    "abstained": ("Not found", "gray", "The documents don't contain an answer."),
    "retrieval_only": (
        "Retrieval only",
        "blue",
        "No LLM configured or selected: showing the most relevant passages.",
    ),
}

INGEST_ICONS = {"ingested": "✅", "replaced": "🔄", "skipped_duplicate": "⏭️", "failed": "❌"}

_CITATION = re.compile(r"\[(S\d+)\]")
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|$~])")


PROVIDER_LABELS = {"anthropic": "Anthropic", "openai": "OpenAI"}


def detect_provider(api_key: str) -> str | None:
    """Same rule as the server (kept here so the UI imports no backend code):
    ``sk-ant-`` -> Anthropic; any other ``sk-`` -> OpenAI."""
    key = api_key.strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-"):
        return "openai"
    return None


def mask_key(api_key: str) -> str:
    """'sk-ant-api03-abcdef…wxyz' style preview that never shows the secret part."""
    key = api_key.strip()
    return f"{key[:7]}…{key[-4:]}" if len(key) > 14 else "•" * len(key)


def status_badge(status: str) -> tuple[str, str, str]:
    return STATUS.get(status, (status, "gray", ""))


def md_escape(text: str) -> str:
    """Escape document text so Markdown/LaTeX in it renders literally."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def highlight_citations(answer: str) -> str:
    """Render inline [S1] markers as highlighted chips; the rest is escaped text."""
    parts = _CITATION.split(answer)
    out = []
    for i, part in enumerate(parts):
        out.append(f":blue-background[{part}]" if i % 2 else md_escape(part))
    return "".join(out)


def page_label(page: int | None) -> str:
    return f"p. {page}" if page else ""


def source_label(source: str, page: int | None) -> str:
    return f"{source} · {page_label(page)}" if page else source


def passage_rows(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, p in enumerate(passages, start=1):
        chunk = p["chunk"]
        rows.append(
            {
                "rank": rank,
                "source": chunk["source"],
                "page": chunk.get("page"),
                "score": _round(p.get("score")),
                "dense": _round(p.get("dense_score")),
                "bm25": _round(p.get("lexical_score")),
                "rerank": _round(p.get("rerank_score")),
                "chunk_id": chunk["chunk_id"],
            }
        )
    return rows


def timing_series(trace: dict[str, Any]) -> dict[str, float]:
    return {t["stage"]: t["duration_ms"] for t in trace.get("timings", [])}


def ingest_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "": INGEST_ICONS.get(r["status"], ""),
            "file": r["source"],
            "status": r["status"].replace("_", " "),
            "chunks": r.get("num_chunks") or None,
            "details": "; ".join(filter(None, [r.get("message"), *r.get("warnings", [])])),
        }
        for r in report.get("results", [])
    ]


def ingest_summary(report: dict[str, Any]) -> tuple[int, int, int]:
    """(indexed, skipped, failed) counts."""
    results = report.get("results", [])
    ok = sum(r["status"] in ("ingested", "replaced") for r in results)
    skipped = sum(r["status"] == "skipped_duplicate" for r in results)
    return ok, skipped, len(results) - ok - skipped


def document_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "file": d["source"],
            "type": d["file_type"],
            "pages": d.get("num_pages"),
            "chunks": d["num_chunks"],
            "size (KB)": round(d["size_bytes"] / 1024, 1),
            "ingested": str(d.get("ingested_at", ""))[:19].replace("T", " "),
            "doc_id": d["doc_id"],
        }
        for d in documents
    ]


def retrieval_metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for r in report.get("retrieval", []):
        m = r["metrics"]
        rows.append(
            {
                "mode": r["mode"],
                "reranker": r["reranker"],
                "hit@1": _round(m.get("hit@1")),
                "hit@3": _round(m.get("hit@3")),
                "hit@5": _round(m.get("hit@5")),
                "MRR@10": _round(m.get("mrr@10")),
                "p50 ms": _round(m.get("latency_p50_ms"), 1),
            }
        )
    return rows


def category_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """hit@5 per question category, one row per mode/reranker configuration."""
    return [
        {
            "config": f"{r['mode']} / {r['reranker']}",
            **{k: _round(v) for k, v in r["by_category"].items()},
        }
        for r in report.get("retrieval", [])
    ]


def answer_metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = (report.get("answers") or {}).get("metrics", {})
    return [{"metric": k, "value": _round(v)} for k, v in metrics.items()]


def _round(value: Any, digits: int = 3) -> Any:
    return round(value, digits) if isinstance(value, float) else value
