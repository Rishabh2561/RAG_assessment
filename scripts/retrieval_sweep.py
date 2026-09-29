"""Retrieval experiment: chunk size x retrieval mode x reranker on the sample corpus.

Reproduces the table in docs/06_EVALUATION.md. Needs no API key (retrieval only).

    python scripts/retrieval_sweep.py [--chunk-sizes 400,700,1000] [--rerank]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from rag_generator.config import Settings
from rag_generator.evaluation import EvaluationRunner, load_dataset
from rag_generator.observability import configure_logging
from rag_generator.orchestration import RAGApplication
from rag_generator.orchestration.factory import build_reranker

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "sample" / "northwind"
DATASET = ROOT / "data" / "eval" / "northwind_eval.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-sizes", default="400,700,1000")
    parser.add_argument("--modes", default="dense,bm25,hybrid")
    parser.add_argument("--rerank", action="store_true", help="Also run the cross-encoder.")
    parser.add_argument("--out", type=Path, default=ROOT / "eval_reports" / "retrieval_sweep.json")
    args = parser.parse_args()
    configure_logging("WARNING")

    items = load_dataset(DATASET)
    rerankers = ["none", "cross_encoder"] if args.rerank else ["none"]
    rows = []
    with tempfile.TemporaryDirectory() as data_dir:
        for size in [int(s) for s in args.chunk_sizes.split(",")]:
            settings = Settings(
                data_dir=Path(data_dir),
                chunk_size=size,
                chunk_overlap=min(150, size // 5),
                llm_provider="none",
                _env_file=None,
            )
            app = RAGApplication(settings)
            name = f"sweep_c{size}"
            app.ingestion().ingest_paths([CORPUS], name)
            n_chunks = len(app.repository.open(name).store)
            runner = EvaluationRunner(app, name)
            for reranker_name in rerankers:
                reranker = build_reranker(settings.model_copy(update={"reranker": reranker_name}))
                for mode in args.modes.split(","):
                    report = runner.evaluate_retrieval(items, mode, reranker)
                    rows.append(
                        {
                            "chunk_size": size,
                            "chunks": n_chunks,
                            "mode": mode,
                            "reranker": reranker_name,
                            **report.metrics,
                            "by_category_hit@3": _hit3_by_category(report.items),
                            "calibration": report.calibration,
                        }
                    )

    print("| chunk size | chunks | mode | reranker | hit@1 | hit@3 | hit@5 | MRR@10 | p50 ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['chunk_size']} | {r['chunks']} | {r['mode']} | {r['reranker']} | "
            f"{r['hit@1']:.3f} | {r['hit@3']:.3f} | {r['hit@5']:.3f} | {r['mrr@10']:.3f} | "
            f"{r['latency_p50_ms']:.1f} |"
        )
    print("\nhit@3 by category:")
    for r in rows:
        cats = ", ".join(f"{k}={v:.2f}" for k, v in r["by_category_hit@3"].items())
        print(f"  c{r['chunk_size']} {r['mode']:<6} {r['reranker']:<13} {cats}")
    for r in rows:
        if r["calibration"]:
            print(f"\ncalibration c{r['chunk_size']} {r['mode']}: {r['calibration']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _hit3_by_category(items: list[dict]) -> dict[str, float]:
    categories = sorted({i["category"] for i in items})
    return {
        c: sum(i["hit@3"] for i in items if i["category"] == c)
        / max(1, sum(1 for i in items if i["category"] == c))
        for c in categories
    }


if __name__ == "__main__":
    main()
