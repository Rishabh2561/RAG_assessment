# 04 — Implementation Plan

The work is delivered in phases, each ending with passing tests. Phases 2–8 build
the deterministic baseline. Only then are optional components (reranker, rewrite)
evaluated. Status reflects the final state of the repository.

| Phase | Name | Status |
|-------|------|--------|
| 1 | Project foundation | ✅ |
| 2 | Document ingestion (parsers) | ✅ |
| 3 | Chunking | ✅ |
| 4 | Embeddings | ✅ |
| 5 | Vector storage + catalog | ✅ |
| 6 | Retrieval (dense, BM25, hybrid) + reranking | ✅ |
| 7 | Generation | ✅ |
| 8 | Citations + grounding | ✅ |
| 9 | Orchestration + interfaces (CLI, API) | ✅ |
| 10 | Evaluation | ✅ |
| 11 | Observability | ✅ |
| 12 | Testing hardening + final polish | ✅ |

---

### Phase 1 — Project foundation
- **Goal:** A reproducible, installable package skeleton with typed config.
- **Tasks:** Write `pyproject.toml` (src layout, `rag` console script, `[dev]`
  extras) and a `uv.lock`. Add `.gitignore` and `.env.example`. Build `Settings`
  with cross-field validation, a shared error hierarchy (`errors.py`), and domain
  models (`models/`).
- **Expected output:** `uv sync` succeeds; `rag --help` runs.
- **Tests:** `tests/unit/test_config.py`: defaults, env override, invalid overlap,
  unknown provider, secret not in repr.
- **Definition of done:** Config tests green; no hard-coded model names or paths
  outside `Settings` defaults.

### Phase 2 — Document ingestion
- **Goal:** Turn a file into a `ParsedDocument` with page-aware sections, or a typed
  error.
- **Tasks:** Define the `DocumentParser` protocol and `ParserRegistry`. Implement
  `PdfParser` (PyMuPDF, with scanned-page detection), `TextParser` (with an encoding
  fallback) and `DocxParser`. Raise `UnsupportedFileTypeError`, `EmptyDocumentError`,
  `CorruptDocumentError` and `OCRRequiredError`.
- **Expected output:** A parser for each supported format, plus a registry keyed by
  extension.
- **Tests:** Parse generated PDF/DOCX/TXT/MD fixtures. Cover empty file, corrupt PDF,
  image-only PDF, unknown extension, and latin-1 text.
- **Definition of done:** Every failure produces a typed error with a human-readable
  message.

### Phase 3 — Chunking
- **Goal:** Deterministic, page-bounded, overlapping chunks.
- **Tasks:** Implement `RecursiveChunker` with a separator hierarchy, greedy merge,
  overlap, and minimum-size filtering. Make chunk IDs deterministic.
- **Expected output:** `list[Chunk]` with `doc_id`, `source`, `page`, `index` and
  `text`.
- **Tests:** Size bound, overlap present, no cross-page chunks, determinism, tiny
  input, a text with no separators.
- **Definition of done:** Property-style tests pass for several sizes.

### Phase 4 — Embeddings
- **Goal:** Pluggable text → normalised vectors.
- **Tasks:** Define the `EmbeddingProvider` protocol. Implement `FastEmbedProvider`
  (lazy load, batching, query prefix) and `HashingEmbeddingProvider` (deterministic,
  offline). Wrap errors in `EmbeddingError`.
- **Tests:** Hashing: determinism, unit norm, similar texts closer than dissimilar
  ones. FastEmbed: an opt-in slow test (`-m slow`).
- **Definition of done:** The test suite needs no model download.

### Phase 5 — Vector storage
- **Goal:** Persistent per-collection store with exact search.
- **Tasks:** Define the `VectorStore` protocol. Implement `NumpyVectorStore` (add,
  delete by document, search, atomic persist/load, manifest with the embedding
  model) and `DocumentCatalog`.
- **Tests:** Round-trip persist/load, delete, search ordering, dimension mismatch,
  model mismatch → `IndexMismatchError`.
- **Definition of done:** Reloading a collection yields identical search results.

### Phase 6 — Retrieval (+ reranking)
- **Goal:** Dense, BM25 and hybrid retrieval behind one interface; optional
  reranker.
- **Tasks:** Implement the BM25 index (tokenizer, Okapi scoring), `DenseRetriever`,
  `BM25Retriever`, `HybridRetriever` (RRF), `NoOpReranker` and
  `CrossEncoderReranker`.
- **Tests:** BM25 ranks an exact-term document first; RRF fuses correctly (known
  ranks → known order); hybrid rescues a lexical-only match; the dense score is
  preserved for the gate.
- **Definition of done:** All three modes are selectable by config.

### Phase 7 — Generation
- **Goal:** Structured grounded answers from a provider-neutral interface.
- **Tasks:** Define the `LLMProvider` protocol and `LLMResponse`. Write the prompt
  templates (system contract; sources delimited as untrusted data). Implement
  `AnthropicProvider` (structured output, typed error mapping, usage capture,
  optional server-side fallback).
- **Tests:** Prompt rendering (source numbering, page labels, context budget); the
  Anthropic provider with a mocked client (success, refusal, rate limit →
  `GenerationError`, malformed JSON).
- **Definition of done:** No vendor import outside `anthropic_provider.py` and the
  factory.

### Phase 8 — Citations and grounding
- **Goal:** Checkable claim-to-source links and explicit abstention.
- **Tasks:** Validate and resolve citations. Derive `grounding_status`. Add the
  relevance gate. Extract inline `[S#]` markers as a fallback when the structured
  list is empty.
- **Tests:** Invalid IDs dropped; inline markers parsed; gate abstains below the
  threshold; status transitions.
- **Definition of done:** Every answer carries a status and resolved citations.

### Phase 9 — Orchestration and interfaces
- **Goal:** End-to-end ingest and query via CLI and API.
- **Tasks:** Build `IngestionService` (hash-based idempotency, same-name
  replacement, per-file isolation), `QueryService` (flow, optional corrective retry,
  trace), the factory, the `typer` CLI and the FastAPI app (upload via temporary file,
  error → HTTP status mapping).
- **Tests:** Integration: ingest → query with hashing embeddings and `FakeLLM`;
  duplicate skip; replacement; the rewrite path. E2E: CLI via `CliRunner`, API via
  `TestClient`.
- **Definition of done:** Two different sample document sets work in separate
  collections without code changes.

### Phase 10 — Evaluation
- **Goal:** Metrics that can drive decisions.
- **Tasks:** Define the dataset schema (JSONL) and a sample dataset over
  `data/sample`. Implement retrieval metrics (hit@k, recall@k, MRR), answer metrics
  (abstention accuracy, citation validity, citation hit, keyword recall) and an
  optional LLM judge. Add `rag eval` with a mode comparison and JSON report output.
- **Tests:** Metric functions against hand-computed values; the runner with fakes.
- **Definition of done:** A real retrieval evaluation has been run with bge-small,
  and its results are recorded in `06_EVALUATION.md` and the decision log.

### Phase 11 — Observability
- **Goal:** Every stage is traceable without leaking content.
- **Tasks:** JSON log formatter, `stage_timer` context manager, per-query
  `QueryTrace` (returned by the API and printed by `rag ask --trace`), and an opt-in
  `RAG_LOG_CONTENT`.
- **Tests:** Log records contain the expected fields; no chunk text appears unless
  enabled.

### Phase 12 — Hardening and polish
- **Goal:** Assessment-ready repository.
- **Tasks:** Full suite, lint (ruff), code-quality review, architecture review,
  docs 05–10, README, and the final self-review.
- **Definition of done:** The checklist in the README is satisfied.
