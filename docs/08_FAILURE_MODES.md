# 08 — Failure Modes

For each failure: how it's **detected**, how it's **handled**, what the **user
experiences**, and what is **logged**. Log events are structured (see
[10_OBSERVABILITY.md](10_OBSERVABILITY.md)). Document text is never logged by default.

## Ingestion

| Failure | Detection | Handling | User experience | Logging |
|---------|-----------|----------|-----------------|---------|
| **Empty document** | No text after whitespace normalisation, or the chunker produces 0 chunks | `EmptyDocumentError`; that file is marked `failed`, the batch continues | `[failed] notes.txt: notes.txt: file contains no text` | `document_failed error_type=EmptyDocumentError` (WARNING) |
| **Corrupt document** | PyMuPDF / python-docx / python-pptx / openpyxl raises on open; password-protected PDF or Office file | `CorruptDocumentError` with the underlying reason | `[failed] x.pdf: not a readable PDF (…)` | `document_failed error_type=CorruptDocumentError` |
| **Unsupported file** | Extension not in the registry | `UnsupportedFileTypeError` listing supported types. Directory walks skip such files and count them | Explicit file: `[failed] …: unsupported file type '.xls'. Supported: …`. In a directory: skipped | `unsupported_files_skipped count=N` |
| **OCR-required (scanned) PDF** | Page has < 20 characters of text *and* contains images | Whole document scanned → `OCRRequiredError`. Some pages scanned → those pages are skipped with a warning | `[failed] scan.pdf: …appears to be scanned images. OCR is not supported.` or `warning: 3 page(s) look scanned…` | `document_failed error_type=OCRRequiredError`; warnings in `IngestResult` |
| **File too large** | `len(bytes) > RAG_MAX_FILE_MB`; the API reads at most limit + 1 bytes | `FileTooLargeError` for that file | `[failed] big.pdf: 72.0 MB exceeds the 50 MB limit` | `document_failed error_type=FileTooLargeError` |
| **Duplicate document** | SHA-256 of the bytes already in the catalog (even under another name) | Skipped; nothing re-embedded | `[skipped_duplicate] handbook.pdf` | `document_skipped_duplicate doc_id=…` |
| **Updated document (same name, new content)** | `catalog.find_by_source(name)` returns a different `doc_id` | Old chunks and catalog entry removed; new version indexed | `[replaced] policy.txt: 4 chunks` | `document_ingested status=replaced` |
| **Conflicting documents** | Not detectable at ingestion; both are valid data | Both indexed. The prompt instructs the model to surface disagreement and cite both sides | The answer states both values with both citations (eval q17: 8 h vs 9.5 h) | Citations list both chunk IDs |
| **Embedding failure** (download/load/inference) | Exception inside `FastEmbedProvider` | `EmbeddingError`; the file is marked `failed`. Load retried on the next call | `[failed] …: could not load embedding model '…' … Check the model name and network access` | `document_failed error_type=EmbeddingError` |
| **Embedding model changed since index was built** | `manifest.model_id != configured model_id` | `IndexMismatchError` on add or search | "collection was built with embedding model 'A' but the configured model is 'B'. Re-ingest…" (HTTP 409) | Error log |
| **Crash mid-persist** | — | Temp file + `os.replace` (atomic). Index written before catalog; orphan chunks dropped on open | Previous snapshot intact; at worst the last document must be re-ingested | `orphan_chunks_dropped chunks=N` |
| **Unreadable / inconsistent index** | `np.load` fails, schema version differs, or chunk count ≠ vector count | `RAGError` telling you to re-ingest | Clear error; never silently served | Error log |

## Query

| Failure | Detection | Handling | User experience | Logging |
|---------|-----------|----------|-----------------|---------|
| **Invalid question** | Empty, whitespace only, or > `RAG_MAX_QUESTION_CHARS` | `InvalidQueryError` | HTTP 400 / CLI error | — |
| **Collection missing or empty** | Repository lookup; `len(store) == 0` | `CollectionNotFoundError` / `CollectionEmptyError` | HTTP 404 / 409 | — |
| **No relevant context (off-topic)** | Best dense cosine < `RAG_MIN_RELEVANCE` (0.50) | Abstain **without calling the LLM** | "I couldn't find an answer to that in the provided documents." + reason with the score and threshold | `query_completed status=abstained attempts=0 best_dense=…` |
| **No relevant context (on-topic, answer absent)** | LLM returns `answerable=false` | Abstain with the LLM's reason. The gate can't catch this: scores overlap with answerable questions (06 §4.2) | Same message, reason e.g. "The sources do not mention the CEO." | `status=abstained attempts=1` |
| **Retrieval returns bad context** | Not detectable per query without labels; detected in aggregate by `rag eval` (hit@k) | Runtime: the LLM should abstain rather than guess. Optional `RAG_QUERY_REWRITE` retries with rephrased queries. Offline: tune per 06 §5 | A safe "not found" rather than a wrong answer | `retrieved_chunk_ids`, `best_dense`, `rewrites` |
| **Ambiguous query** | The LLM sees partial or multiple readings | Prompt: answer the supported part and say what is missing. No clarification loop (single-shot) | Partial answer naming the missing piece | — |
| **LLM API timeout** | `APITimeoutError` after SDK retries (`RAG_LLM_MAX_RETRIES`, exponential backoff) | `GenerationError(retryable=True)` | HTTP **503** `{"error":"GenerationError",…,"retryable":true}` | `query_failed` (ERROR) with trace fields |
| **Rate limiting (429)** | `RateLimitError` after SDK retries (honouring `retry-after`) | `GenerationError(retryable=True)` | HTTP 503, `retryable: true` | `query_failed` |
| **LLM 5xx / network failure** | `APIStatusError ≥ 500` / `APIConnectionError` | `GenerationError(retryable=True)` | HTTP 503 | `query_failed` |
| **No credentials configured** | SDK raises `TypeError` ("Could not resolve authentication method") or `CredentialsError` before sending | `GenerationError(retryable=False)` (D13) | `Error: no Anthropic credentials found; set ANTHROPIC_API_KEY (environment or .env), or run retrieval-only with --no-llm / RAG_LLM_PROVIDER=none` (exit 1 / HTTP 503) | `query_failed` |
| **OpenAI: no credentials / content filter** | `OpenAI()` raises `OpenAIError` ("Missing credentials"); `finish_reason == "content_filter"`; `message.refusal` set | `GenerationError` with guidance to set `OPENAI_API_KEY` / the reason | `Error: no OpenAI credentials found; set OPENAI_API_KEY …` (exit 1 / HTTP 503) | `query_failed` |
| **OpenAI embeddings unavailable** | `OpenAIError` during `embeddings.create` | `EmbeddingError`; each file fails individually | `[failed] …: OpenAI embedding request failed (…) Check OPENAI_API_KEY and RAG_EMBEDDING_MODEL.` | `document_failed error_type=EmbeddingError` |
| **Key entered in the UI is invalid, or its provider can't be detected** | Vendor returns 401 on `/llm/verify` or `/query`; the key isn't `sk-ant-…`/`sk-…` and no provider was chosen | "Check key" reports it without spending tokens on a real answer; undetectable keys → 400 | Sidebar: "Anthropic rejected this API key. Check it and try again." / "Couldn't tell which provider this key is for. Choose it above." | `query_failed`; the key is never logged |
| **Auth / request error (401/400)** | `AuthenticationError` / `BadRequestError` | `GenerationError(retryable=False)` with guidance ("set ANTHROPIC_API_KEY or use RAG_LLM_PROVIDER=none") | HTTP 503, `retryable: false` | `query_failed` |
| **Model refusal** | `stop_reason == "refusal"` (after server-side fallback, if enabled) | `GenerationError` | "the model declined to answer this request" | `query_failed` |
| **Truncated output** | `stop_reason == "max_tokens"` | `GenerationError` naming `RAG_LLM_MAX_TOKENS` | Error message | `query_failed` |
| **Malformed output** | JSON decode or schema validation fails | `GenerationError`; rare with structured outputs | Error; never shown as an answer | `query_failed` |
| **Hallucination** | Runtime: mitigated by the grounding prompt, abstention path and citation validation. Offline: `--judge` faithfulness and false-answer rate | Answers must cite; uncited answers are `unverified` | `grounding_status` tells the user how much to trust the answer | Status and citation IDs |
| **Citation mismatch** | Cited ID ∉ supplied source map | Invalid IDs dropped and inline markers stripped; if none remain, status `unverified` | Answer shown with valid citations only | `invalid_citations_dropped ids=[…]` (WARNING); `trace.invalid_citations` |
| **Context overflow** | Sum of passage lengths > `RAG_MAX_CONTEXT_CHARS` | Lower-ranked passages dropped; top passage always kept | Transparent | Only sources actually sent can be cited |
| **Prompt injection in a document** | See [09_SECURITY_AND_PRIVACY.md](09_SECURITY_AND_PRIVACY.md) | Delimiting, escaping, system instruction to treat sources as data | The answer should ignore injected instructions | — |
