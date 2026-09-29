# Decision Log

A chronological record of significant decisions, including ones revised after
evidence. The stable rationale for each accepted decision lives in
[02_ARCHITECTURE_DECISIONS.md](02_ARCHITECTURE_DECISIONS.md). This log records *when*
and *why* things were decided or changed.

---

### 2026-09-30 — D1: Treat the task prompt's brief summary as the source brief
- **Context:** No brief file was attached or present in the repository.
- **Options:** Stop and ask for the file / proceed with the summarised brief.
- **Decision:** Proceed. Record the brief verbatim in `docs/BRIEF.md` and tag
  requirements by origin (SOURCE / SPEC / ASSUMPTION / DECISION).
- **Reason:** The summary states the core requirements unambiguously. Everything
  else is explicitly left to the candidate.
- **Trade-off:** If the full brief contains extra constraints, they aren't captured.
- **Impact:** `00_REQUIREMENTS.md` legend.

### 2026-09-30 — D2: Environment probe before choosing the stack
- **Context:** The machine runs Python 3.14 (new), and no LLM API key is available.
- **Options:** Assume wheels exist / verify.
- **Decision:** Test-installed every candidate dependency in a throwaway venv, and
  test-ran fastembed embedding and reranking models.
- **Reason:** Native wheels (onnxruntime, faiss, PyMuPDF) are the most common
  "works on my machine" failure.
- **Result:** All candidates install on 3.14. bge-small loads in about 20 s the first
  time and embeds in milliseconds. The MiniLM cross-encoder works.
- **Impact:** Local fastembed is viable (ADR-003). The test suite must not need a key
  or a model download (N5), which leads to `HashingEmbeddingProvider` and `FakeLLM`.

### 2026-09-30 — D3: NumPy exact search instead of FAISS or Chroma
- **Context:** FAISS installed fine, so this wasn't a forced choice.
- **Decision:** NumPy (ADR-004).
- **Reason:** At the target scale FAISS would use `IndexFlatIP`, which is the same
  computation, while still needing a metadata side store. Chroma duplicates our
  abstractions.
- **Trade-off:** No ANN and no concurrent writers.
- **Impact:** `retrieval/vector_store.py`. pgvector or FAISS stays a one-class swap.

### 2026-09-30 — D4: Retrieval-only mode (`RAG_LLM_PROVIDER=none`)
- **Context:** Evaluators may not have a key, and we had none during development.
- **Decision:** A first-class mode that returns ranked passages with no generated
  answer.
- **Reason:** It keeps the system demonstrable and the retrieval half of evaluation
  runnable by anyone.
- **Trade-off:** One more code path to test.
- **Impact:** `QueryService`, CLI output, API schema (`answer` is nullable).

### 2026-09-30 — D5: Structured JSON output instead of provider-native citations
- **Context:** Anthropic offers native citations, but they are incompatible with
  structured output and are vendor-specific.
- **Decision:** JSON-schema output `{answerable, answer, citations}` with `[S#]` IDs
  (ADR-010).
- **Reason:** Vendor neutrality, and a machine-checkable abstention flag.
- **Trade-off:** Citations are ID-level, not span-level.

### 2026-09-30 — D6: Default model `claude-opus-5-5` at effort `low`, with server-side refusal fallback
- **Context:** A model had to be chosen. Grounded QA over about 1.5k tokens of
  supplied context doesn't need deep reasoning.
- **Decision:** Use the current default Opus model with `effort=low` to keep latency
  and cost down, and enable `fallbacks: "default"` (beta
  `server-side-fallback-2026-07-01`) so a classifier decline on a benign document
  gets re-served instead of failing. Both are configurable (`RAG_LLM_MODEL`,
  `RAG_LLM_EFFORT`, `RAG_ANTHROPIC_SERVER_FALLBACK`).
- **Trade-off:** Opus costs more per token than Sonnet or Haiku. Cheaper tiers are one
  env var away, and the evaluation harness is how you'd confirm parity before
  switching.
- **Note:** The server-side fallback beta works only on the first-party Claude API.
  Set it to `false` for Bedrock, Vertex, or a proxy.

<!-- Entries below were added during and after implementation. -->

### 2026-09-30 — D7: Parsers take bytes, not paths
- **Context:** HTTP uploads would otherwise have to be written to temporary files before
  parsing.
- **Options:** Temporary file on disk / parse from memory.
- **Decision:** `DocumentParser.parse(data: bytes, source: str)`. PyMuPDF
  (`stream=`) and python-docx (`BytesIO`) both support this.
- **Reason:** Uploaded documents never touch disk except in the index itself. That
  removes a whole class of temp-file cleanup and leakage issues (09_SECURITY).
- **Trade-off:** The whole file is held in memory. Bounded by `RAG_MAX_FILE_MB`
  (50 MB by default).

### 2026-09-30 — D8: Default retrieval mode revised from `hybrid` to `dense` (evidence-driven)
- **Context:** ADR-005 initially defaulted to hybrid on the prior that BM25 rescues
  exact-identifier queries. The retrieval sweep (`scripts/retrieval_sweep.py`; 26
  questions, 21 of them answerable, 6 documents) measured:

  | chunk | mode | hit@1 | hit@3 | hit@5 | MRR |
  |---|---|---|---|---|---|
  | 700 | dense | 0.905 | 0.952 | 1.000 | 0.933 |
  | 700 | bm25 | 0.714 | 0.857 | 0.857 | 0.770 |
  | 700 | hybrid | 0.810 | 0.905 | 0.905 | 0.854 |

  Exact-ID questions scored hit@3 = 1.00 in *every* mode, so BM25 added nothing there.
  Paraphrase questions were hurt: for q10 ("fully recharge"), BM25 has no match
  because it doesn't stem ("recharge" vs "charging"). Under equal-weight RRF, chunks
  present in both lists then outrank a dense-only hit, pushing the right chunk out of
  the top 10.
- **Options:** Keep hybrid / switch to dense / weighted fusion / add stemming to BM25.
- **Decision:** Default `RAG_RETRIEVAL_MODE=dense`. Hybrid and BM25 stay implemented
  and selectable.
- **Reason:** It's measured-better or equal on every category here, and it's the
  simpler pipeline.
- **Trade-off:** The sample is small (a one-question difference at hit@3), so this
  is directional, not significant. Corpora with many near-identical identifiers (part
  numbers, ticket IDs, legal clause numbers) are where BM25 should earn its place.
  Switch to `hybrid` there and re-run `rag eval` to confirm.
- **Impact:** `Settings.retrieval_mode`, `.env.example`, ADR-005 status → Revised.
  Possible future work: stemming in the BM25 tokenizer, and weighted RRF.

### 2026-09-30 — D9: Relevance gate calibrated to 0.50 for bge-small, and scoped to off-topic questions only
- **Context:** Best-dense-score distributions (chunk size 700): answerable 0.60–0.86;
  on-topic unanswerable 0.63–0.735 ("CEO?", "list price?", "pension?"); off-topic
  unanswerable 0.47 ("sourdough").
- **Finding:** The distributions **overlap**. No similarity threshold can separate
  on-topic unanswerable questions from answerable ones: a threshold of 0.64 would block
  4 answerable questions.
- **Decision:** `RAG_MIN_RELEVANCE=0.50`, which sits below the lowest answerable
  score (0.593 across all chunk sizes) with margin. The gate is a cheap filter for
  clearly off-topic questions. Abstaining on on-topic questions is the LLM's job,
  through the `answerable` flag.
- **Trade-off:** Most unanswerable questions still cost one LLM call. We choose that
  over false abstentions, which a user experiences as a broken system.
- **Impact:** Settings default. ADR-010 updated.

### 2026-09-30 — D10: Chunk size 700 / overlap 140 (was 1000 / 150)
- **Context:** Sweep over 400, 700 and 1000 characters (dense, no reranker): hit@1 was
  0.905 / 0.905 / 0.810 and MRR 0.940 / 0.933 / 0.877.
- **Decision:** 700 / 140 (20% overlap).
- **Reason:** It matches 400 on ranking quality with 43% fewer chunks, and gives
  larger self-contained passages for multi-part answers than 400 does. Compared with
  1000, citations are more precise and each passage costs fewer tokens.
- **Trade-off:** It's tuned on one corpus. Chunk size is the first knob to revisit on
  a new corpus (06_EVALUATION §6).

### 2026-09-30 — D11: Cross-encoder reranker stays off by default (evidence-driven)
- **Context:** With the cross-encoder at chunk size 1000, dense hit@1 went from 0.81 to
  1.00. At 700 it made no difference (0.905 → 0.905). It adds about 1.5 s p50 latency
  on CPU for 20 candidates.
- **Decision:** Keep `RAG_RERANKER=none`.
- **Reason:** The generator consumes the top 5, and dense hit@5 is already 1.00, so
  the reranker improves ordering *inside* the context window. That matters little to
  answer quality, and the latency cost is large.
- **When to revisit:** If evaluation on a new corpus shows hit@5 well below hit@20
  (right chunk retrieved but ranked out of the context window), enable it.
