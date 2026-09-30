# 02 — Architecture Decision Records

Each ADR follows the format **Context → Decision → Alternatives → Trade-offs →
Consequences**. The comparative scoring across options is in
[03_TRADEOFF_ANALYSIS.md](03_TRADEOFF_ANALYSIS.md), and the chronological record of
when decisions were made or revised is in [DECISION_LOG.md](DECISION_LOG.md).

Status values: *Accepted* (implemented), *Accepted — default off* (implemented but
disabled until evaluation justifies it), *Revised* (changed after evidence).

---

## ADR-001 — Document parsing: PyMuPDF + lightweight native parsers

**Status:** Accepted

**Context.** Documents arrive at runtime in unknown formats. We need reliable text
extraction with **page numbers** (needed for citations), clear detection of files we
can't handle, and an install that works on Windows, macOS and Linux without system
packages.

**Decision.** A `ParserRegistry` maps file extensions to parsers:

- `.pdf` → `PdfParser`, built on **PyMuPDF** (`fitz`), producing one section per page.
- `.txt` and `.md` → `TextParser`, with UTF-8 decoding and a latin-1 fallback.
- `.docx` → `DocxParser`, built on `python-docx`: paragraphs, then tables flattened to
  text.
- `.pptx` → `PptxParser`, built on `python-pptx`: one section per slide, with the slide
  number used as the page. Text boxes, grouped shapes, tables and speaker notes.
- `.xlsx` / `.xlsm` → `XlsxParser`, built on `openpyxl` (read-only, cached formula
  values): one section per non-empty sheet.
- `.csv` / `.tsv` → `CsvParser` (stdlib `csv`, delimiter sniffed).

Spreadsheet rows are flattened to one self-describing line each,
`[Sheet] Row 7: Region: West | Revenue: 1200`, so any chunk cut from a long table
still carries its column names and the row can be found in the source file. Legacy
binary `.xls` and `.ppt` are not supported (they would need `xlrd` or a LibreOffice
conversion).

A PDF whose pages yield almost no text is classified as *probably scanned* and
rejected with `OCRRequiredError`, not indexed as empty.

**Alternatives.**
- *Unstructured*: broadest format coverage, layout-aware elements and built-in OCR
  hooks. However, it pulls in a very large dependency tree (and optional system
  dependencies such as poppler and tesseract), is slow on large PDFs, and is more than
  four formats need.
- *pypdf*: pure Python, but its text extraction is weaker on multi-column layouts and
  ligatures, and it's slower.
- *pdfplumber*: excellent for tables, slower, and table structure is out of scope.
- *Docling / LlamaParse / cloud document AI*: best layout fidelity, but either heavy
  ML models or external calls that send documents to third parties.

**Trade-offs.** We accept weaker layout understanding (tables are flattened, reading
order in complex layouts follows PyMuPDF's block order) and no OCR, in exchange for
a fast, dependency-light, cross-platform parser with exact page numbers. PyMuPDF is
AGPL-licensed; that's fine for an assessment, but it would need a commercial licence,
or a switch to pypdf, in a closed-source product.

**Consequences.** Adding a format means adding one class and one registry entry. If
the evaluation corpus turns out to be table-heavy or scanned, the parser is the
component to replace (Unstructured or Docling behind the same interface).

---

## ADR-002 — Chunking: recursive character splitter with page-bounded chunks

**Status:** Accepted

**Context.** Chunk boundaries directly affect retrieval recall and citation
precision. Document structure varies (prose, lists, headings), and the chunker must
work on any document set without tuning per corpus.

**Decision.** A self-implemented `RecursiveChunker`:

- Target size of **700 characters** with **140 characters of overlap**, both
  configurable. The initial 1,000 / 150 was revised by measurement (DECISION_LOG D10).
- Split first on blank lines, then newlines, then sentence ends, then spaces, then
  characters, and greedily merge the pieces back up to the target size.
- Chunk **within** a section (a page, for PDFs), never across one, so every chunk
  maps to exactly one page.
- Chunks shorter than `RAG_MIN_CHUNK_CHARS` (default 20) are dropped as noise, such
  as page numbers and headers.

**Alternatives.**
- *Fixed-size windows*: simplest, but they cut mid-sentence and mid-word, which harms
  both embeddings and readability of citations.
- *Semantic chunking* (split where embedding similarity between adjacent sentences
  drops): can produce more coherent chunks, but it costs an embedding call per
  sentence at ingest time, needs a corpus-dependent threshold, and published
  comparisons show inconsistent gains over recursive splitting.
- *Structure-aware (Markdown headers / DOM)*: great when structure exists, but it
  degrades to fixed-size on flat text, and each format needs its own rules.
- *LangChain's `RecursiveCharacterTextSplitter`*: the same algorithm, but it brings in
  a large framework for about 60 lines of logic.

**Trade-offs.** Characters are a proxy for tokens (about 4 characters per token for
English, so 700 characters is roughly 175 tokens, well inside bge-small's 512-token
window). We accept imprecision on non-English text for zero tokenizer dependency.
Page-bounded chunks can make a chunk small when a paragraph spans a page break; we
accept that for exact page citations.

**Consequences.** Chunk size and overlap are the first knobs to turn if evaluation
shows poor recall (see [06_EVALUATION.md](06_EVALUATION.md) §6).

---

## ADR-003 — Embeddings: local fastembed (bge-small-en-v1.5) behind `EmbeddingProvider`

**Status:** Accepted

**Context.** Every chunk and every query must be embedded. The system must run
without an API key for tests and retrieval-only use, and the embedding model must be
swappable without touching retrieval.

**Decision.** The default is `FastEmbedProvider` with `BAAI/bge-small-en-v1.5`
(384-dim, ~130 MB ONNX, runs on CPU, no PyTorch), with the query instruction prefix
applied only to queries. A second implementation, `HashingEmbeddingProvider`
(feature-hashed word uni- and bigrams), is deterministic, needs no download, and is
used by the test suite. It is explicitly **not** a quality embedder.

**Alternatives.**
- *Hosted API* (OpenAI `text-embedding-3-*`, Voyage, Cohere): higher quality on some
  benchmarks and no local compute. However, it needs another API key, sends every
  document to a third party, adds network latency and cost to ingestion, and ties the
  index to a vendor.
- *sentence-transformers* (PyTorch): the same models and more choice, but a roughly
  2 GB PyTorch install for the same inference.
- *Larger local models* (bge-base or bge-large, e5-large): better recall, but 3–10×
  slower ingestion on CPU.

**Trade-offs.** A small local model gives up some retrieval quality compared with
the best hosted models, in exchange for privacy, zero marginal cost, offline
operation and a modest install. The first run downloads the model (about 20 s on our
link). Because the manifest records the model, changing it requires re-ingestion,
and the store refuses mixed vectors (`IndexMismatchError`).

**Consequences.** Hybrid retrieval (ADR-005) compensates for the small dense model's
weakness on exact identifiers. If retrieval recall is the bottleneck, a larger model
is a one-variable change.

---

## ADR-004 — Vector store: in-process NumPy exact search behind `VectorStore`

**Status:** Accepted

**Context.** We need to persist per-collection vectors plus chunk metadata, run
cosine search, delete by document, and run with zero infrastructure. Expected scale is
≤ 10⁵ chunks per collection.

**Decision.** `NumpyVectorStore` keeps a `float32` matrix in memory and computes
exact cosine similarity with one matrix-vector product. It persists to `vectors.npy`
plus `chunks.jsonl` with atomic writes.

**Alternatives.**
- *FAISS*: fast approximate search at millions of vectors, but it has no metadata
  storage (you still need a side store for chunks), delete support varies by index
  type, it's a native dependency, and at 10⁵ × 384 the exact `IndexFlatIP` it would
  use is the same computation as NumPy.
- *Chroma*: an embedded store with metadata filtering and persistence built in, but it
  brings a heavier dependency (SQLite, its own embedding-function abstraction, and
  telemetry defaults), and its abstractions overlap with ours.
- *pgvector*: a real database with transactions, metadata joins, access control and
  concurrent writers, but it requires running PostgreSQL, which is disproportionate
  for a single-process assessment.
- *Qdrant, Weaviate, Milvus*: production vector DBs with filtering and replication,
  but each is another service to operate.

**Trade-offs.** For a single-process assessment, exact NumPy search minimises
operational complexity and gives *exact*, not approximate, results. At 10⁵ chunks,
a query is about 40 MB of `float32` and single-digit milliseconds. The trade-off is
no concurrent writers, the whole collection in RAM, and O(n) search. If the system
needed multi-user access control, transactional metadata or horizontal deployment,
pgvector (or Qdrant) would become more appropriate. Above about 10⁶ chunks, FAISS HNSW
or IVF would be the natural next step.

**Consequences.** The `VectorStore` protocol is deliberately small (`add`,
`delete_document`, `search`, `get`, `all_chunks`, `persist`), so a pgvector or FAISS
implementation is a single new class plus a factory entry.

---

## ADR-005 — Retrieval: dense by default; BM25 and hybrid (RRF) selectable

**Status:** Revised. The initial default was `hybrid`. Measurement changed it to
`dense` (DECISION_LOG D8).

**Context.** Unknown corpora contain both paraphrasable prose (where dense retrieval
shines) and exact tokens such as product codes, error codes, section numbers and
names (where dense retrieval with a small model is weak).

**Decision.** Implement three retrievers behind one interface: `DenseRetriever`,
`BM25Retriever` (an in-house Okapi BM25, about 50 lines, built from the stored chunks
at load time), and `HybridRetriever`, which fuses the two ranked lists with
**reciprocal-rank fusion** (RRF, k = 60). `RAG_RETRIEVAL_MODE` selects the mode.

The design-time default was hybrid. The retrieval sweep showed dense ≥ hybrid on
every question category of the sample corpus, including exact identifiers. Hybrid
*lost* a paraphrase question, because BM25 (no stemming) missed it and RRF then
promoted chunks found by both retrievers above the dense-only hit. **The default is
therefore `dense`.** Hybrid remains a one-variable switch for identifier-heavy
corpora.

**Alternatives.**
- *Dense only*: the simplest option, but it misses exact-match queries.
- *BM25 only*: no model needed and strong on keywords, but it fails on paraphrase and
  synonyms.
- *Weighted score fusion* (α·dense + (1−α)·bm25): needs score normalisation and a
  tuned α that is corpus-dependent. RRF is rank-based and parameter-light.
- *`rank-bm25` package*: works, but it is unmaintained and trivial to implement, and
  owning the tokenizer keeps BM25 consistent with our text handling.

**Trade-offs.** Hybrid costs one extra in-memory scoring pass (sub-millisecond at our
scale) and a little code. RRF discards score magnitudes, so we keep the dense cosine
of each candidate separately for the relevance gate (ADR-010).

**Consequences.** The evaluation harness runs all three modes on the same dataset so
the default is chosen from measured results, not preference. That is exactly what
happened here. The known weaknesses of our hybrid are the unstemmed BM25 and the
equal-weight RRF. Stemming and weighted RRF are the documented next steps if a
corpus needs lexical matching.

---

## ADR-006 — Reranking: implemented, **off by default**

**Status:** Accepted — default off

**Context.** Cross-encoders often improve precision at the top ranks, but they add
latency (one model forward pass per candidate) and another model download.

**Evidence (DECISION_LOG D11).** At chunk size 1000, the cross-encoder raised dense
hit@1 from 0.81 to 1.00. At 700 it changed nothing. It costs about 1.5 s p50 on CPU,
and dense hit@5 is already 1.00, so for a generator that reads the top 5 it
reorders passages the LLM sees anyway.

**Decision.** The `Reranker` interface has two implementations: `NoOpReranker`
(the default) and `CrossEncoderReranker` (fastembed
`Xenova/ms-marco-MiniLM-L-6-v2`, about 80 MB ONNX). When enabled, retrieval fetches
`RAG_RERANK_CANDIDATES` (default 20) and the reranker keeps `RAG_TOP_K`.

**Alternatives.**
- *No reranker at all*: the simplest, but it gives no way to test the hypothesis that
  ranking is the bottleneck.
- *LLM-as-reranker*: high quality, but an extra paid LLM call per query.
- *Hosted rerank API* (Cohere, Voyage): high quality, but another vendor and key, and
  documents leave the machine.

**Trade-offs.** Off by default because it only pays off when relevant chunks are
retrieved but mis-ordered. The evaluation measures exactly this (recall@20 compared
with MRR). Turning it on adds tens to hundreds of milliseconds on CPU.

**Consequences.** Evaluation decides. See [06_EVALUATION.md](06_EVALUATION.md) §6.

---

## ADR-007 — LLM: Claude via the Anthropic SDK behind `LLMProvider`; retrieval-only mode when unset

**Status:** Accepted

**Context.** Answer generation needs strong instruction-following (answer *only* from
the sources, cite them, abstain otherwise), reliable structured output, and a long
enough context window for top-k chunks.

**Decision.**
- `AnthropicProvider` uses the official `anthropic` SDK.
- The default model is `claude-opus-5-5`, configurable through `RAG_LLM_MODEL`.
  Sampling parameters are not sent by default, because current Opus and Sonnet models
  reject them. `RAG_LLM_TEMPERATURE` is sent only when explicitly set, for models that
  accept it.
- The default effort is `low`, because grounded QA over supplied context is not a
  deep-reasoning task.
- Structured output uses a JSON schema (`output_config.format`), so the response is
  always parseable into `{answerable, answer, citations[]}`.
- Server-side refusal fallback (`fallbacks: "default"`) is enabled by default on the
  first-party API and can be switched off.
- `RAG_LLM_PROVIDER=none` gives a **retrieval-only** mode that returns ranked
  passages with no generated answer, so the system is usable and inspectable without
  a key.

**Alternatives.**
- *Other hosted models* (OpenAI, Gemini): comparable capability, different SDK. The
  `LLMProvider` interface makes this one class.
- *Local model via Ollama or llama.cpp*: private and free per call, but a large
  download, it needs a GPU for acceptable latency, and small local models follow the
  "cite or abstain" contract noticeably less reliably.
- *A cheaper Claude tier* (`claude-sonnet-5-5`, `claude-haiku-4-5`): lower cost and
  latency. It is a config change, and a sensible choice for high-volume use once
  evaluation confirms answer quality holds.

**Trade-offs.** A hosted frontier model gives the best adherence to the grounding
contract at a per-query cost, and sends retrieved passages (not whole documents) to a
third party. The top-k context (about 1.5k tokens) keeps cost per query small.

**Consequences.** Vendor code is confined to `generation/anthropic_provider.py`.
Tests use a scripted `FakeLLM`.

---

## ADR-014 — OpenAI as a second provider for generation and (opt-in) embeddings

**Status:** Accepted

**Context.** Users may already hold an OpenAI key rather than an Anthropic one, or
need a model hosted on Azure OpenAI or an OpenAI-compatible server (Ollama, vLLM).
The `LLMProvider` and `EmbeddingProvider` interfaces were designed for exactly this
swap. This ADR records how the second vendor was added.

**Decision.**
- `RAG_LLM_PROVIDER=openai` → `OpenAIProvider` (official `openai` SDK). It uses Chat
  Completions **strict JSON-schema structured outputs**, so the same prompts, answer
  schema, citation validation and abstention logic run unchanged. Our schemas already
  meet strict mode's rules (`additionalProperties: false`, every property required),
  and a test asserts this.
- `RAG_EMBEDDING_PROVIDER=openai` → `OpenAIEmbeddingProvider`
  (`text-embedding-3-small`). This is **opt-in**: it sends document text to OpenAI at
  ingest time, which reverses ADR-003's privacy property.
- Model names default **per provider** (`claude-opus-5-5` / `gpt-5.5`;
  `BAAI/bge-small-en-v1.5` / `text-embedding-3-small`) when `RAG_LLM_MODEL` /
  `RAG_EMBEDDING_MODEL` are unset. `RAG_OPENAI_BASE_URL` enables Azure and compatible
  servers.
- `reasoning_effort` and `temperature` are sent only when configured, because each is
  rejected by one family of models (reasoning vs non-reasoning).
- The `openai` package is an optional extra (`.[openai]`). The factory imports it
  lazily and gives an install hint if it is missing.

**Alternatives.**
- *OpenAI Responses API* instead of Chat Completions: a newer surface, but Chat
  Completions is what OpenAI-compatible servers implement, so one code path covers
  OpenAI, Azure and Ollama.
- *A generic multi-vendor layer (LiteLLM, LangChain)*: many vendors for free, but it
  adds a large dependency and hides request details (structured-output support varies
  by backend). Two thin providers are about 120 lines each.
- *Tool calling to force JSON*: works on older models, but strict structured outputs
  is the direct mechanism.

**Trade-offs.**
- **The relevance gate is embedding-model-specific.** The 0.50 threshold was
  calibrated for bge-small only (D9). OpenAI embeddings produce differently scaled
  cosine similarities, so for any uncalibrated model the gate defaults to **off**,
  with a start-up warning, until `rag eval` suggests a value. The alternative, reusing
  0.50 blindly, could silently reject answerable questions, which is the worse
  failure.
- The default OpenAI model (`gpt-5.5`) was chosen from the SDK's model list, and
  answer quality with it has **not been measured** (no key during development). Run
  `rag eval --answers` before relying on it.
- Switching embedding provider requires re-ingesting. The index refuses mixed vector
  spaces (`IndexMismatchError`).

**Consequences.** The vendor boundary still holds: `openai` is imported only in
`generation/openai_provider.py`, `embeddings/openai_provider.py` and (lazily) the
factory. Adding a third vendor follows the same pattern.

---

## ADR-008 — Orchestration: plain Python services, no framework

**Status:** Accepted

**Context.** The pipeline is a short, mostly linear sequence with one optional loop.

**Decision.** Two plain Python services (`IngestionService`, `QueryService`) wired by
a factory, with constructor dependency injection of protocol-typed components.

**Alternatives.**
- *LangChain*: many ready-made integrations, but heavy abstraction layers, frequent
  API churn, harder debugging (deep stack traces and hidden prompts), and we would use
  about 5% of it.
- *LangGraph*: an explicit state graph that is excellent for multi-step agents with
  branching, checkpoints and human-in-the-loop, but our graph has one conditional edge.
- *LlamaIndex*: an opinionated RAG framework with rapid prototyping, but it obscures
  exactly the decisions this assessment is meant to demonstrate.

**Trade-offs.** We write about 150 lines of glue that a framework would provide, and
in return get full visibility into prompts, retries and data flow, easy testing with
fakes, and a small dependency surface.

**Consequences.** If the agentic behaviour grows (tool selection, multi-hop
decomposition, human approval), LangGraph becomes the preferable host, and the
existing components drop into its nodes unchanged.

---

## ADR-009 — Query rewriting / agentic behaviour: one bounded corrective retry, off by default

**Status:** Accepted — default off

**Context.** One real failure mode of RAG is *vocabulary mismatch*: the user asks
"can I work from abroad?" and the document says "remote work outside the country of
employment". Retrieval misses, and the model correctly abstains, but the answer
existed. Agentic loops are often added without addressing a specific failure.

**Decision.** When `RAG_QUERY_REWRITE=true` and context is judged insufficient (the
LLM returns `answerable=false`, or the relevance gate fails), the `QueryService` asks the LLM for up to three
alternative phrasings of the query. It retrieves again, merges the results with the
first round (deduplicating by chunk ID), and generates **once** more. Maximum one
retry. The context-sufficiency check is folded into the generation call's structured
output, so the happy path costs exactly one LLM call.

**Alternatives.**
- *No rewriting*: the simplest option and predictable latency, but vocabulary-mismatch
  misses are unrecoverable.
- *Always rewrite* (HyDE or multi-query on every request): improves recall on some
  benchmarks, but adds an LLM call to *every* query, including the majority that
  don't need it.
- *Separate grader call before generation* (CRAG-style): a more explicit signal, but
  it adds an extra call on the happy path.
- *Open-ended agent loop with a retrieval tool*: flexible, but its cost and latency
  are unbounded, it's harder to evaluate, and it isn't justified by any requirement.

**Trade-offs.** Correctly unanswerable questions cost about 3× (generate, rewrite,
generate). Answerable questions cost the same as the baseline. Off by default so the
baseline is measured first. Enable it when evaluation shows abstentions on questions
labelled answerable (false abstentions).

**Consequences.** The trace records `rewritten_queries` and `attempts`, so the effect
is observable per query and measurable in evaluation.

---

## ADR-010 — Citations and grounding: numbered source IDs, validated after generation, plus a relevance gate

**Status:** Accepted

**Context.** "Grounded" must be verifiable. Free-text references ("according to the
handbook…") can't be checked. Models sometimes cite sources they weren't given.

**Decision.**
1. Each retrieved chunk is rendered as `[S1]…[Sk]` with its file name and page.
2. The prompt requires inline `[S#]` markers and a structured `citations` list, and
   requires `answerable=false` with a short reason when the sources don't contain the
   answer.
3. **Post-validation:** citation IDs not in the supplied set are dropped and logged.
   Each valid ID is resolved to `{doc_id, source, page, chunk_id, snippet}`.
4. `grounding_status` is:
   - `grounded`: answerable with ≥ 1 valid citation.
   - `unverified`: answerable but no valid citation. The answer is returned with a
     warning flag.
   - `abstained`: not answerable.
5. **Relevance gate:** if the best dense cosine among retrieved chunks is below
   `RAG_MIN_RELEVANCE` (0.50 for bge-small), abstain without calling the LLM.
   Calibration showed on-topic unanswerable questions score as high as answerable
   ones (0.63–0.74 vs 0.60–0.86), so the gate is deliberately set to catch only
   *off-topic* questions. Abstaining on on-topic questions is the LLM's job
   (DECISION_LOG D9).

**Alternatives.**
- *Anthropic native citations* (document or search-result blocks): character-exact
  cited spans, but vendor-specific, and incompatible with structured JSON output.
- *No citations, only a source list*: simpler, but it can't tell which claim came from
  where.
- *Post-hoc NLI or entailment verification of each sentence*: stronger faithfulness
  guarantees, but another model and more latency. We do this in evaluation (LLM
  judge), not at runtime.

**Trade-offs.** ID-level citations prove *which* passage the model claims to use,
not that the passage entails the claim. Entailment is measured offline. The relevance
threshold is embedding-model-specific and must be recalibrated if the model changes.
That is documented next to the setting.

**Consequences.** Citation validity and citation hit rate are first-class evaluation
metrics.

---

## ADR-011 — Configuration and dependency management: pydantic-settings + uv + pyproject

**Status:** Accepted

**Context.** No value that varies by environment (keys, model names, sizes, paths)
may be hard-coded. Evaluators need a reproducible install.

**Decision.**
- A single `Settings` class (`pydantic-settings`) with the `RAG_` prefix, loaded from
  environment variables and an optional `.env`.
- Cross-field validation (for example `overlap < size`, `top_k ≤ rerank_candidates`).
- `ANTHROPIC_API_KEY` is read under the SDK's standard name, and the value is stored
  as `SecretStr`, so it can't appear in reprs or logs.
- A `pyproject.toml` (PEP 621) with a `uv.lock` for exact reproducibility. `pip
  install -e .` works too.
- Optional extras (`[dev]`) keep the runtime install lean.

**Alternatives.**
- *YAML or TOML config file*: nice for nested settings, but secrets would then need a
  separate mechanism anyway, and env vars are the twelve-factor norm for containers.
- *Hydra*: powerful for experiment sweeps, but overkill here.
- *Poetry*: mature, but slower, and `uv` handles both lock and venv.
- *requirements.txt*: universal, but it has no lock of transitive dependencies.

**Trade-offs.** Flat env-var config is less expressive for deeply nested settings. We
have about 25 settings, so flat is fine.

---

## ADR-012 — Interfaces: CLI + REST API, plus a Streamlit UI over the API

**Status:** Revised. The original decision was CLI + REST API with no custom UI. A
Streamlit UI was added on request (DECISION_LOG D15); see "Revision" below.

**Context.** Users must supply documents at runtime and ask questions. Evaluators need
a fast way to try the system.

**Decision.** A `typer` CLI (`rag ingest | ask | docs | delete | drop | eval | serve`)
and a FastAPI app. FastAPI's auto-generated Swagger UI at `/docs` provides
browser-based file upload and querying with no front-end code.

**Alternatives.**
- *Streamlit or Gradio UI*: nicer visuals, but another dependency, and UI code that
  doesn't demonstrate RAG engineering.
- *Notebook*: good for exploration, but poor as an application.

**Trade-offs.** Less polished UX than a dedicated UI, but a much smaller surface area.
Both interfaces are thin adapters over the same services, so adding a UI later is
additive.

**Revision: Streamlit UI (`rag ui`).** The UI is a **pure HTTP client** of the REST
API. It imports no backend code: `rag_generator/ui/client.py` is its only way in.
- *Why over REST rather than in-process:* one backend process serves the CLI, Swagger,
  the UI and any other client. The UI can run on another machine. And the API
  contract (`/config`, `/evaluate`, the error bodies) is exercised by a real client.
- *What it cost:* three API additions, useful beyond the UI:
  - `GET /config`: supported file types, limits, active models and defaults, so the
    UI hard-codes none of them.
  - `POST /collections/{c}/evaluate`: the evaluation harness over HTTP.
  - A per-request `rewrite` flag on `/query`.
  - Also, the CLI and API now share one `run_evaluation()` instead of duplicating it.
- *Alternatives:*
  - Streamlit in-process: one process, but it couples the UI to the backend's Python
    environment and models.
  - Gradio: similar, with less layout control for the evaluation dashboard.
  - A React SPA: the most polished, but a second toolchain.
- *Trade-off accepted:* two processes to run. `rag ui --with-api` starts both.
  Like the API, the UI has **no authentication**, so it binds to 127.0.0.1 by default.

---

## ADR-013 — Evaluation: offline labelled dataset, retrieval metrics without an LLM, answer metrics with one

**Status:** Accepted

**Context.** Architectural choices (chunk size, retrieval mode, reranker, rewrite)
should be driven by measurement. Evaluation must be runnable without an API key for
the retrieval half.

**Decision.** A JSONL dataset of questions, each labelled with expected source
documents (and optionally pages), expected answer keywords, and an `answerable` flag.
`rag eval` computes:

- Retrieval metrics per mode: hit@k, recall@k, MRR.
- If an LLM is configured, answer metrics: abstention accuracy, citation validity,
  citation hit rate, keyword recall, and an optional LLM-judge faithfulness score.
- Latency percentiles and token usage.

**Alternatives.**
- *RAGAS or DeepEval*: ready-made metrics, but heavy dependencies and LLM-judged by
  default, which makes them costly and non-deterministic.
- *Manual spot checks only*: cheap, but not repeatable.

**Trade-offs.** Keyword recall is a crude proxy for answer correctness, so we pair it
with an optional judge. A small hand-labelled dataset gives directional signal, not
statistical significance.
