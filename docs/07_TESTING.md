# 07 — Testing

## 1. Strategy

| Layer | Location | Dependencies | Speed |
|-------|----------|--------------|-------|
| Unit | `tests/unit/` | None beyond the package (fakes, in-memory PDF/DOCX builders) | < 1 s |
| Integration | `tests/integration/` | Real parsers, chunker, store and retrieval. `HashingEmbeddingProvider` and a scripted `FakeLLM` | ~ 3 s |
| End-to-end | `tests/e2e/` | CLI via `typer.testing.CliRunner`; API via `fastapi.testclient.TestClient` | ~ 5 s |
| Slow (opt-in) | `-m slow` | Real `bge-small-en-v1.5` model (downloads about 130 MB once) | ~ 10 s |

**Design choice: the default suite is offline and deterministic** (requirement N5).
Two test doubles make this possible:

- `HashingEmbeddingProvider` is a real, deterministic embedder: feature-hashed word
  n-grams. It exercises every code path that handles vectors, and ships as a
  selectable provider.
- `FakeLLM` (in `tests/conftest.py`) is a scripted `LLMProvider` whose responses are
  functions of the prompt. It records every call, so tests can assert *how many* LLM
  calls were made. That matters for the relevance gate and for the bounded rewrite
  retry.

The Anthropic provider is tested against a mocked SDK client, using real
`anthropic` exception classes, so the error mapping is verified without network
access.

Run:

```bash
pytest                  # offline suite
pytest -m slow          # + real-model retrieval regression (needs network once)
ruff check src tests scripts && ruff format --check src tests scripts
```

## 2. Test inventory (representative cases)

### Unit

| File | Representative cases |
|------|----------------------|
| `test_config.py` | Defaults are valid. Env vars and `.env` override defaults. `overlap >= size` is rejected. `top_k > candidate_pool` is rejected. Unknown provider or mode is rejected. A path-traversal collection name is rejected. **The API key never appears in `repr` or `model_dump_json`.** |
| `test_parsers.py` | A PDF yields one section per page with page numbers. Markdown whitespace is normalised. Latin-1 fallback works with a warning. DOCX tables are flattened. The whole sample corpus parses. Empty txt/md → `EmptyDocumentError`. Corrupt PDF or DOCX → `CorruptDocumentError`. Image-only PDF → `OCRRequiredError`. Blank PDF → empty. `.xlsx`, `.png` and extensionless files → `UnsupportedFileTypeError`, listing the supported types. |
| `test_chunker.py` | The size bound holds for three size/overlap pairs. Consecutive chunks overlap. Paragraph boundaries are preferred. Text with no separators is hard-split. **Chunks never cross pages.** IDs are deterministic and sequential. Tiny fragments are dropped. Invalid overlap is rejected. |
| `test_embeddings.py` | Hashing is deterministic, unit-norm and `float32`. Similar texts score higher. `model_id` distinguishes vector spaces. *(slow)* The real model ranks the paraphrase above an unrelated sentence. |
| `test_vector_store.py` | Cosine ordering. `k` larger than the store. Delete bumps `version`. Persist/reload round-trip leaves no temp files. **Model mismatch → `IndexMismatchError` on search and on add.** Dimension mismatch. Empty store. **Orphan chunks (crash between index and catalog writes) are dropped on open.** |
| `test_retrieval.py` | BM25 ranks the exact term first. Stop-words or unknown terms give no hits. Empty corpus. **RRF order matches hand-computed 1/(k+rank) sums**, and component scores are carried over. A single-list item. |
| `test_prompts_and_citations.py` | Sources are numbered and page-labelled. The question comes after the context. The context budget is respected and the first source is always kept. **Document text can't close `</source>` or inject `<question>` (it is escaped).** Inline `[S#]` markers are extracted in order. Invalid IDs are dropped, and inline and declared citations are merged. Invalid markers are stripped. |
| `test_anthropic_provider.py` | Structured output and usage are parsed, and thinking blocks are ignored. The fallback beta is sent by default and can be disabled. Temperature is sent only when configured. `refusal` and `max_tokens` stop reasons raise. Invalid JSON raises. **Six SDK exception types map to `GenerationError` with the correct `retryable` flag.** |
| `test_metrics.py` | hit@k, RR, source recall, keyword recall, percentile and mean against hand-computed values. The best threshold separates cleanly when possible and prefers the lower threshold when classes overlap. |
| `test_logging.py` | JSON formatter fields. **Content is redacted by default** and shown only with `log_content=True`. `log_event` attaches fields. `StageTimer`. |

### Integration

| File | Representative cases |
|------|----------------------|
| `test_ingestion.py` | Ingesting the sample directory (PDF page count recorded). **Re-ingesting identical content is a no-op.** A changed file with the same name replaces the old one (one catalog entry, new text only). **A mixed batch of good, empty, scanned, corrupt and unsupported files: each bad file fails individually and good files are indexed.** Size limit. **Two collections are isolated and survive a "restart".** Delete. The directory walk skips unsupported and hidden files. **An embedding-model change is detected.** |
| `test_query_service.py` | Grounded answer with resolved citations. PDF citations carry the page. LLM abstention. **Invalid citation IDs are dropped → `unverified`.** **Gate abstains with zero LLM calls.** Retrieval-only mode. **Rewrite path: exactly 3 LLM calls (answer, rewrite, answer) and `attempts == 2`.** Rewrite bounded to one retry. Rewrite is off by default (1 call). LLM failure → retryable `GenerationError`. Schema-violating output raises. Empty, whitespace or oversized questions. Missing or empty collection. Every retrieval mode finds an exact identifier. |
| `test_review_regressions.py` | One test per defect found in the independent code review (DECISION_LOG D12): **a failed replace keeps the old version**; ghost catalog entries are reconciled; same base names in different directories stay distinct; duplicate names in one batch are rejected; a renamed duplicate leaves no stale version; an unexpected parser exception is isolated; the BM25 cache is never stale after a concurrent mutation; a dropped collection refuses writes from a stale handle; a rewrite failure falls back to abstention; the gate triggers the rewrite; an empty rewrite skips the retry; schema errors don't leak model output. |
| `test_evaluation.py` | **Every label in the shipped dataset is satisfiable by the corpus.** Retrieval report metrics. Answer metrics with a fake LLM, including that `false_answer_rate` catches an always-answering model. |

### End-to-end

| File | Representative cases |
|------|----------------------|
| `test_cli.py` | **Two different document sets in two collections, configured only via env vars (acceptance criterion AC1).** docs/delete/drop. `--trace` output. Errors print a message without a traceback. Invalid config exits 2 naming the variable. An all-failed batch exits 1. `rag eval` writes a JSON report. |
| `test_api.py` | Upload then query with citations (passages omitted by default). `retrieval_only` and `include_passages`. List and delete documents, drop a collection. **Upload filenames are sanitised (`../../etc/evil.txt` → `evil.txt`).** Oversized upload rejected per file. **Error → HTTP status mapping** (404, 422, 400). **LLM outage → 503 with `retryable: true`.** Health. |
| `test_real_models.py` *(slow)* | Default configuration on the sample corpus: hit@5 ≥ 0.95, MRR ≥ 0.85, and **no answerable question falls below the gate threshold**. |

## 3. Failure tests mapped to failure modes

| Failure mode | Test |
|--------------|------|
| Empty / corrupt / unsupported / scanned file | `test_parsers.py`, `test_ingestion.py::test_bad_files_fail_individually…` |
| Oversized file | `test_ingestion.py::test_file_size_limit`, `test_api.py::test_oversized_upload…` |
| Duplicate document | `test_ingestion.py::test_reingesting_identical_content_is_a_noop` |
| Changed document with the same name | `test_ingestion.py::test_changed_file_with_same_name…` |
| No relevant context | `test_query_service.py::test_relevance_gate_abstains…`, `…test_llm_abstention…` |
| Embedding model changed | `test_vector_store.py::test_model_mismatch…`, `test_ingestion.py::test_embedding_model_change…` |
| LLM failure, rate limit or timeout | `test_anthropic_provider.py::test_sdk_errors…`, `test_api.py::test_llm_outage…` |
| Malformed LLM output | `test_anthropic_provider.py::test_invalid_json…`, `test_query_service.py::test_schema_violating…` |
| Citation mismatch | `test_query_service.py::test_invalid_citations…` |
| Crash between writes | `test_vector_store.py::test_repository_drops_orphan_chunks` |
| Prompt-injection delimiter escape | `test_prompts_and_citations.py::test_document_text_cannot_break_out…` |
| Invalid configuration | `test_config.py`, `test_cli.py::test_invalid_config…` |

## 4. Known gaps

- **Live LLM behaviour** (the model's actual abstention and citation discipline) is
  not unit-testable. It is measured by `rag eval --answers --judge`, which needs a key
  and wasn't run during development (06 §4.3).
- **Concurrency:** there is no stress test of concurrent upload and query in the API.
  Correctness relies on the per-collection write lock and on the vector store swapping
  its state atomically (08).
- **Very large files and corpora:** there are no performance tests beyond the sample
  corpus.
