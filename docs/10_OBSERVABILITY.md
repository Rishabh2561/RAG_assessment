# 10 — Observability

## 1. Mechanisms

| Mechanism | What it gives you | Where |
|-----------|-------------------|-------|
| **Structured log events** | One event per significant step, with typed fields. `RAG_LOG_FORMAT=json` for machines, `text` for humans | `observability/logging.py` (`log_event`) |
| **`QueryTrace`** | A per-query record returned with every answer (API JSON, `rag ask --trace` / `--json`) | `models/results.py` |
| **Stage timings** | `StageTimer` records the duration of every pipeline stage | Included in logs and in the trace |
| **Evaluation reports** | Aggregate quality, latency and token metrics as JSON | `rag eval` → `eval_reports/` |

Logs go to stderr, so `--json` output on stdout stays machine-readable.

## 2. What is logged

### Ingestion

| Event | Fields |
|-------|--------|
| `embedding_model_loaded` | `model`, `load_ms` (separated out so the first file's `embed_ms` isn't misleading) |
| `document_ingested` | `collection`, `source`, `doc_id`, `status` (ingested / replaced), `chunks`, `chars`, `parse_ms`, `chunk_ms`, `embed_ms` |
| `document_skipped_duplicate` | `source`, `doc_id` |
| `document_failed` (WARNING) | `source`, `error_type`, `error` |
| `unsupported_files_skipped` | `directory`, `count` |
| `ingest_batch_completed` | `collection`, `files`, `succeeded`, `failed`, `total_chunks`, `duration_ms` |
| `document_deleted`, `collection_dropped` | `collection`, `doc_id` |
| `orphan_chunks_dropped`, `ghost_catalog_entries_dropped` | `collection`, count (crash recovery) |

### Query

`query_completed` (INFO) or `query_failed` (ERROR) is emitted once per question:

| Field | Meaning |
|-------|---------|
| `collection`, `mode`, `reranker`, `model` | Configuration actually used |
| `status` | `grounded` / `unverified` / `abstained` / `retrieval_only` |
| `chunk_ids` | Retrieved chunk IDs in rank order (what the LLM saw) |
| `best_dense` | Best cosine score, which is the relevance-gate input |
| `citations` | Chunk IDs actually cited |
| `attempts`, `rewrites` | Generation calls and rewritten queries (agentic path) |
| `input_tokens`, `output_tokens` | Token usage, summed over all LLM calls for the question |
| `retrieve_ms`, `generate_ms`, `rewrite_ms`, `retrieve_rewritten_ms`, `generate_retry_ms`, `total_ms` | Stage latencies (only the stages that ran) |
| `question` | `<redacted len=N>` unless `RAG_LOG_CONTENT=true` |
| `error` | On `query_failed` only |

Other query events: `invalid_citations_dropped` (WARNING, with `ids`) and
`rewrite_failed` (WARNING).

Illustrative shape (`RAG_LOG_FORMAT=json`; values are examples, not from a recorded run):

```json
{"ts": "2026-09-30T10:12:03.118+00:00", "level": "INFO",
 "logger": "rag_generator.orchestration.query_service", "event": "query_completed",
 "status": "grounded", "citations": ["6466d60c43db-00002"], "question": "<redacted len=22>",
 "collection": "northwind", "mode": "dense", "reranker": "none", "model": "claude-opus-5-5",
 "attempts": 1, "rewrites": 0,
 "chunk_ids": ["6466d60c43db-00002", "8b1cf9916111-00000", "..."], "best_dense": 0.651,
 "input_tokens": 1412, "output_tokens": 96, "total_ms": 2143.7,
 "retrieve_ms": 11.8, "generate_ms": 2129.4}
```

## 3. What is deliberately *not* logged

- Document text, chunk text and snippets.
- Question text (unless opted in).
- Generated answers.
- API keys (they are `SecretStr` and never formatted).

Chunk IDs are enough to investigate: `rag docs` maps a `doc_id` prefix to the file,
and the chunk index locates the passage.

## 4. Using it to answer operational questions

| Question | Where to look |
|----------|---------------|
| "Why did it say *not found*?" | `best_dense` vs `gate_threshold` (a gate abstention has `attempts=0`), or the LLM's `reason` in the answer |
| "Which passages did the answer use?" | `citations` against `chunk_ids` |
| "Is retrieval working at all?" | `rag eval` hit@k. Per query, `chunk_ids` and `best_dense` |
| "Is the model citing things it wasn't given?" | `invalid_citations_dropped` warnings; the `citation_validity` metric |
| "Where is the latency?" | The stage timings (typically `generate_ms` ≫ `retrieve_ms`) |
| "What does this cost?" | Token fields per query; averages in `rag eval --answers` |
| "Did rewriting help?" | `attempts`, `rewrites` and `rewrite_failed`; compare eval runs with and without `RAG_QUERY_REWRITE` |
| "Why did ingestion skip my file?" | `document_failed` `error_type`, or `unsupported_files_skipped` |

## 5. Not implemented (next steps)

- Distributed tracing (OpenTelemetry spans per stage). `StageTimer` maps one-to-one
  onto spans.
- A metrics endpoint (Prometheus counters and histograms for latency, tokens and
  abstention rate).
- Request IDs propagated from the API into log events, and the Anthropic
  `request_id` logged on failures.
