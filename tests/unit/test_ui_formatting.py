from rag_generator.ui.formatting import (
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


def test_status_badges_cover_every_grounding_status():
    for status in ("grounded", "unverified", "abstained", "retrieval_only"):
        label, colour, explanation = status_badge(status)
        assert label and colour and explanation
    assert status_badge("new_status")[0] == "new_status"  # unknown statuses still render


def test_citation_markers_become_chips_and_text_is_escaped():
    rendered = highlight_citations("Costs $5 *now* [S1] and later [S12].")
    assert ":blue-background[S1]" in rendered and ":blue-background[S12]" in rendered
    assert "\\$5" in rendered and "\\*now\\*" in rendered  # no LaTeX/emphasis injection


def test_md_escape_neutralises_markdown_from_documents():
    assert md_escape("# Title [link](x)") == "\\# Title \\[link\\]\\(x\\)"


def test_source_label_with_and_without_page():
    assert source_label("manual.pdf", 3) == "manual.pdf · p. 3"
    assert source_label("notes.md", None) == "notes.md"


def test_passage_rows_round_scores_and_keep_rank():
    rows = passage_rows(
        [
            {
                "chunk": {"source": "a.pdf", "page": 2, "chunk_id": "c1"},
                "score": 0.123456,
                "dense_score": 0.5,
                "lexical_score": None,
                "rerank_score": None,
            }
        ]
    )
    assert rows == [
        {
            "rank": 1,
            "source": "a.pdf",
            "page": 2,
            "score": 0.123,
            "dense": 0.5,
            "bm25": None,
            "rerank": None,
            "chunk_id": "c1",
        }
    ]


def test_timing_series():
    trace = {
        "timings": [
            {"stage": "retrieve", "duration_ms": 12.5},
            {"stage": "generate", "duration_ms": 900},
        ]
    }
    assert timing_series(trace) == {"retrieve": 12.5, "generate": 900}


def test_ingest_rows_and_summary():
    report = {
        "results": [
            {
                "source": "a.pdf",
                "status": "ingested",
                "num_chunks": 4,
                "message": None,
                "warnings": [],
            },
            {
                "source": "b.pdf",
                "status": "skipped_duplicate",
                "num_chunks": 0,
                "message": "same content as a.pdf",
                "warnings": [],
            },
            {
                "source": "c.pdf",
                "status": "failed",
                "num_chunks": 0,
                "message": "not a readable PDF",
                "warnings": [],
            },
        ]
    }
    assert ingest_summary(report) == (1, 1, 1)
    rows = ingest_rows(report)
    assert [r[""] for r in rows] == ["✅", "⏭️", "❌"]
    assert rows[1]["status"] == "skipped duplicate" and rows[2]["details"] == "not a readable PDF"


def test_document_rows():
    rows = document_rows(
        [
            {
                "source": "a.pdf",
                "file_type": "pdf",
                "num_pages": 4,
                "num_chunks": 7,
                "size_bytes": 2048,
                "ingested_at": "2026-09-30T10:11:12.123+00:00",
                "doc_id": "abc",
            }
        ]
    )
    assert rows[0]["size (KB)"] == 2.0 and rows[0]["ingested"] == "2026-09-30 10:11:12"


def test_metric_rows_tolerate_undefined_metrics():
    report = {
        "retrieval": [
            {
                "mode": "dense",
                "reranker": "none",
                "by_category": {"factual": 1.0},
                "metrics": {
                    "hit@1": 0.9047,
                    "hit@3": None,
                    "hit@5": 1.0,
                    "mrr@10": 0.93333,
                    "latency_p50_ms": 11.26,
                },
            }
        ]
    }
    rows = retrieval_metric_rows(report)
    assert rows[0]["hit@1"] == 0.905 and rows[0]["hit@3"] is None and rows[0]["p50 ms"] == 11.3
    assert category_rows(report) == [{"config": "dense / none", "factual": 1.0}]


def test_ui_key_helpers():
    from rag_generator.ui.formatting import detect_provider, mask_key
    from rag_generator.ui.llm_settings import explain_key_error

    assert detect_provider("sk-ant-api03-x") == "anthropic"
    assert detect_provider("sk-proj-x") == "openai" and detect_provider("xyz") is None
    masked = mask_key("sk-ant-api03-SECRETSECRET-9999")
    assert masked == "sk-ant-…9999" and "SECRET" not in masked
    assert "SECRET" not in mask_key("short-SECRET")
    assert explain_key_error("OpenAI authentication failed; check OPENAI_API_KEY", "openai") == (
        "OpenAI rejected this API key. Check it and try again."
    )
