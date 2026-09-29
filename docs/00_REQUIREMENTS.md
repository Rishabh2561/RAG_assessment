# 00 — Requirements

## Legend

Every item in this document is tagged with its origin so that nothing invented is
presented as coming from the assessment.

| Tag | Meaning |
|-----|---------|
| **SOURCE** | Stated in the candidate brief ([BRIEF.md](BRIEF.md)). Non-negotiable. |
| **SPEC** | Stated in the candidate's own engineering specification (the task prompt around the brief). Binding for this project, but self-imposed, not an assessment requirement. |
| **ASSUMPTION** | Not stated anywhere; a reasonable interpretation we adopted so work could proceed. Each one can be revisited. |
| **DECISION** | An engineering choice we made. Rationale lives in [02_ARCHITECTURE_DECISIONS.md](02_ARCHITECTURE_DECISIONS.md). |

## 1. Problem statement

Build a *RAG generator*: a system that is handed an arbitrary set of documents **at
runtime**, turns them into a searchable knowledge base, and answers natural-language
questions about them with answers that are **grounded** in, and traceable to, those
documents. The same code must serve any document set — a policy handbook today, a set
of research papers tomorrow — with no source changes. (SOURCE)

## 2. Explicit requirements from the brief

| ID | Requirement | Tag |
|----|-------------|-----|
| S1 | Accept documents at runtime. | SOURCE |
| S2 | Create a RAG application over those documents. | SOURCE |
| S3 | Allow users to ask questions and receive grounded answers. | SOURCE |
| S4 | Work with different document sets without code changes. | SOURCE |
| S5 | Architecture, stack, models, tools and approach are the candidate's choice. | SOURCE |
| S6 | Deliver a Git repository containing the working code. | SOURCE |
| S7 | Deliver complete AI-agent transcripts. | SOURCE |
| S8 | Demonstrate strong architectural reasoning and documentation. | SOURCE |

## 3. Functional requirements

| ID | Requirement | Traces to | Tag |
|----|-------------|-----------|-----|
| F1 | Ingest documents supplied while the application is running, via CLI (file paths or directories) and via HTTP upload. | S1 | DECISION (two entry points) |
| F2 | Support at least PDF, plain text, Markdown and DOCX. | S1, S4 | ASSUMPTION (the brief names no formats; these cover the common cases) |
| F3 | Create or update a persistent index for the ingested documents; re-ingesting an unchanged file is a no-op. | S2 | SPEC + DECISION (idempotency) |
| F4 | Keep separate document sets in separate, named **collections**, so different sets never contaminate each other's answers. | S4 | DECISION |
| F5 | Accept a natural-language question against a collection. | S3 | SOURCE |
| F6 | Retrieve relevant passages from the collection. | S2, S3 | SOURCE (implied by "RAG") |
| F7 | Generate an answer using only the retrieved passages. | S3 ("grounded") | SOURCE |
| F8 | Return citations that identify the document, page (where applicable) and passage behind each claim. | S3 ("grounded") | SPEC |
| F9 | Explicitly decline to answer when the documents don't support an answer, rather than guessing. | S3 ("grounded") | SPEC |
| F10 | List and delete documents in a collection. | S1, S4 | DECISION (needed for a usable runtime lifecycle) |
| F11 | Provide an evaluation mechanism that measures retrieval and answer quality against a labelled dataset. | S8 | SPEC |
| F12 | Operate in a retrieval-only mode when no LLM is configured, so the system is inspectable without an API key. | S6 ("working code") | DECISION |

## 4. Non-functional requirements

| ID | Requirement | Tag |
|----|-------------|-----|
| N1 | **Configurability:** model names, chunk size and overlap, top-k, temperature/effort, paths and API keys come from environment variables or a `.env` file, never from source code. | SPEC |
| N2 | **Replaceability:** parser, chunker, embedder, vector store, retriever, reranker and LLM sit behind interfaces, and swapping one requires no change to the others. | SPEC |
| N3 | **Separation:** ingestion-time and query-time code paths are independent; queries never mutate the index. | SPEC |
| N4 | **Observability:** structured logs for every stage, with document IDs, chunk counts, latencies, retrieved chunk IDs and token usage. Document *content* is not logged by default. | SPEC |
| N5 | **Testability:** the full test suite runs offline and deterministically, with no API key and no model download. | SPEC + DECISION |
| N6 | **Reproducibility:** pinned dependency ranges, a lockfile, and deterministic chunk/document IDs. | SPEC |
| N7 | **Error handling:** every failure mode in [08_FAILURE_MODES.md](08_FAILURE_MODES.md) produces a typed error or a documented degraded result, never an unhandled traceback to the user. | SPEC |
| N8 | **Ease of evaluation:** an evaluator can clone the repo, install dependencies and run a question in a handful of commands. | SPEC |
| N9 | **Proportionality:** no component is added without a documented reason; simplicity is preferred. | SPEC |

## 5. Constraints

| Constraint | Tag |
|------------|-----|
| Development machine: Windows 11, Python 3.14, `uv` available. The project must also run on Linux/macOS and Python ≥ 3.11. | ASSUMPTION |
| No LLM API key was available during development. Generation code is exercised with a test double; live generation needs the evaluator's own `ANTHROPIC_API_KEY`. | ASSUMPTION (observed fact) |
| Single-process, single-user deployment. No external database or service is required to run. | DECISION |
| Assessment-scale corpora: tens to low thousands of documents, up to about 10⁵ chunks per collection. | ASSUMPTION |

## 6. Assumptions

| ID | Assumption | Why it is reasonable | What would change if it is wrong |
|----|------------|----------------------|----------------------------------|
| A1 | "Documents" means text-bearing files (PDF, DOCX, TXT, MD). Scanned PDFs are detected and reported, not OCR'd. | OCR adds a heavy native dependency (Tesseract) for an edge case. | Add an OCR parser behind the existing `DocumentParser` interface. |
| A2 | "At runtime" means documents are supplied after the application is built and started, by a user, not baked into the code or image. | This is the plain reading. | — |
| A3 | "Different document sets" means both sequentially (a different set tomorrow) and concurrently (two sets side by side), which is why we have collections. | Covers both readings at low cost. | Collections could be removed; the default collection still works. |
| A4 | English-language documents are the primary target. | The brief is in English, and small English embedding models are the cheapest well-performing option. | Switch `RAG_EMBEDDING_MODEL` to a multilingual model. No code change is needed. |
| A5 | A single user or trusted operator. There is no authentication or multi-tenant access control. | The brief mentions no users or tenants. | See [09_SECURITY_AND_PRIVACY.md](09_SECURITY_AND_PRIVACY.md) §Tenant isolation. |
| A6 | Answer latency of a few seconds is acceptable, and per-question cost matters more than milliseconds. | This is an interactive Q&A tool, not a real-time system. | Choose a faster model or tier via config. |
| A7 | Corpus size fits comfortably in memory (see Constraints). | Assessment scale. | Swap `NumpyVectorStore` for FAISS or pgvector via the `VectorStore` interface. |

## 7. Ambiguities and how we resolved them

| Ambiguity | Options | Resolution | Tag |
|-----------|---------|------------|-----|
| What interface should users interact through? | Web UI / REST API / CLI / notebook | A CLI for evaluators and scripting, plus a FastAPI REST API whose Swagger UI lets you upload files and ask questions in a browser. No bespoke front-end. | DECISION |
| What counts as "grounded"? | Answer merely uses retrieval / answer cites sources / answer is verified against sources | Every answer must cite retrieved passage IDs. Citations are validated against what was actually retrieved, and the system abstains when evidence is insufficient. Faithfulness is *measured* in evaluation, not guaranteed at runtime. | DECISION |
| Should new uploads replace or extend the index? | Replace / append / upsert | Upsert by content hash: identical content is skipped, and a changed file under the same name replaces its old chunks. | DECISION |
| Is multi-turn chat required? | Single-shot Q&A / conversational memory | Single-shot Q&A. Conversation memory brings query-condensing complexity the brief doesn't ask for. | DECISION (out of scope) |
| Does "agentic" behaviour need to be present? | Mandatory / only where it helps | The brief doesn't require it. We build a deterministic baseline and add one bounded corrective step (query rewrite and retry on insufficient context) behind a flag. | DECISION |
| Which LLM vendor? | Any | Claude (default) or OpenAI, selected by `RAG_LLM_PROVIDER`, each isolated behind `LLMProvider` (ADR-007, ADR-014). | DECISION |

## 8. Out of scope

- OCR for scanned or image-only PDFs (they are detected and reported).
- Images, tables-as-structure, spreadsheets, audio and video.
- Authentication, authorisation and multi-tenant isolation beyond named collections.
- Conversational memory and follow-up question rewriting.
- Horizontal scaling, background job queues and distributed indexes.
- A custom web front-end (Swagger UI and the CLI serve this role).
- Fine-tuning of embedding or generation models.

## 9. Acceptance criteria

| ID | Criterion | How it is verified |
|----|-----------|--------------------|
| AC1 | A document set that has never been seen before can be ingested with a single CLI command or HTTP call, with no code change. | E2E test `tests/e2e/test_cli.py`; README quick start using a second sample set. |
| AC2 | A question returns an answer with ≥ 1 citation that resolves to a real document, page and passage from the retrieved set. | Integration tests on `QueryService` using a fake LLM; citation validation unit tests. |
| AC3 | A question whose answer isn't in the documents returns an explicit "not found in the documents" result with `answerable = false`. | Unit and integration tests; eval abstention metric. |
| AC4 | The embedder, vector store, retriever mode, reranker and LLM can each be changed by configuration alone. | Configuration and factory tests. |
| AC5 | Unsupported, empty, corrupt and scanned files produce clear per-file errors and never abort the rest of the batch. | Failure tests in `tests/unit/test_parsers.py` and `tests/integration/test_ingestion.py`. |
| AC6 | `pytest` passes offline with no API key and no network. | CI-equivalent local run (see [07_TESTING.md](07_TESTING.md)). |
| AC7 | `rag eval` produces retrieval metrics (hit@k, recall@k, MRR) and, when an LLM is configured, answer metrics (abstention accuracy, citation validity, citation hit rate). | Evaluation run output in [06_EVALUATION.md](06_EVALUATION.md). |
| AC8 | Every stage emits a structured log line with its latency; document text is absent from logs unless it is explicitly enabled. | Observability tests; [10_OBSERVABILITY.md](10_OBSERVABILITY.md). |
