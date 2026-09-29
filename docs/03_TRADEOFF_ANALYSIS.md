# 03 — Trade-off Analysis

This document compares options **for this assessment's context**. It doesn't rank
technologies in general. That context:

- It must run on an evaluator's laptop from a fresh clone, with minimal setup.
- It must handle an arbitrary, unknown document set at runtime.
- It runs as a single process for a single user, at ≤ 10⁵ chunks per collection.
- Tests must pass offline with no API key.
- The work is judged on engineering judgement and clarity as much as on raw answer
  quality.

## Decision criteria and weights

| Criterion | Weight | Why it matters here |
|-----------|--------|---------------------|
| Correctness / answer quality | High | This is the product. |
| Operational simplicity (setup, no services) | High | Evaluators must run it quickly (N8). |
| Replaceability / testability | High | An explicit spec requirement (N2, N5). |
| Cost per query / ingest | Medium | Matters, but volumes are small. |
| Latency | Medium | Interactive Q&A; seconds are acceptable (A6). |
| Scalability | Low | Out of scope beyond 10⁵ chunks, but the upgrade path must exist. |

Rating scale: **L** = low, **M** = medium, **H** = high. For *Cost*, *Latency* and
*Complexity*, lower is better. For *Scalability*, higher is better.

## Decision matrix

| Decision | Option | Pros | Cons | Cost | Latency | Complexity | Scalability | Selected |
|----------|--------|------|------|------|---------|------------|-------------|----------|
| **Parsing** | PyMuPDF (+ native txt/md/docx) | Fast; exact page numbers; pip-only install | No OCR; flattened tables; AGPL | L | L | L | M | ✅ |
| | Unstructured | Many formats; layout elements; OCR hooks | Huge dependency tree; system deps; slow | L | H | H | M | |
| | pypdf | Pure Python; permissive licence | Weaker extraction; slower | L | M | L | M | |
| | Docling / cloud document AI | Best layout and table fidelity | Heavy ML models or data leaves the machine | M–H | H | M | H | |
| **Chunking** | Fixed-size | Trivial; predictable | Cuts sentences; worse embeddings | L | L | L | H | |
| | Recursive (page-bounded) | Respects natural boundaries; exact page citations; corpus-agnostic | Char ≈ token proxy; small chunks at page breaks | L | L | L | H | ✅ |
| | Semantic | Coherent topical chunks | Extra embedding pass; corpus-tuned threshold; inconsistent gains | M | M | M | M | |
| | Structure-aware (headers) | Excellent on well-structured Markdown/HTML | Degrades on flat text; per-format rules | L | L | M | H | |
| **Embeddings** | Local fastembed bge-small (ONNX) | Offline; private; free; no PyTorch | Lower ceiling than top hosted models; first-run download | L | L | L | M | ✅ |
| | Hosted API (OpenAI / Voyage / Cohere) | Top quality; no local compute | Key + vendor lock; docs leave machine; per-token cost | M | M | L | H | |
| | Local large (bge-large, e5-large) | Better recall | 3–10× slower CPU ingest; bigger download | L | H | L | M | |
| **Vector store** | NumPy exact search | Zero infrastructure; exact results; ~50 lines | All in RAM; single writer; O(n) | L | L | L | L | ✅ |
| | FAISS | Scales to 10⁷+ (ANN) | Separate metadata store; native dependency; delete semantics vary | L | L | M | H | |
| | Chroma | Metadata + persistence built in | Heavier dependency; overlapping abstractions | L | L | M | M | |
| | pgvector | Transactions, ACLs, SQL joins, concurrency | Requires a Postgres service | M | L | H | H | |
| **Retrieval** | Dense only | Handles paraphrase | Misses exact codes and names with a small model | L | L | L | H | |
| | BM25 only | Exact terms; no model | Fails on paraphrase and synonyms | L | L | L | H | |
| | Hybrid (RRF) | Robust across both query types; parameter-light | Extra scoring pass; some complexity | L | L | M | H | ✅ |
| **Reranking** | None | Zero latency and dependencies | No fix when ranking is the bottleneck | L | L | L | H | ✅ default |
| | Cross-encoder (MiniLM, local) | Better precision at the top ranks | +50–300 ms CPU; another model | L | M | M | M | available |
| | LLM or hosted rerank API | Highest quality | Per-query cost; another vendor | H | H | M | H | |
| **LLM** | Claude (hosted, `claude-opus-5-5`, effort low) | Strong grounding and abstention adherence; structured output | Per-query cost; passages sent to the API | M | M | L | H | ✅ |
| | Cheaper hosted tier (Sonnet 5.5 / Haiku 4.5) | Lower cost and latency | Somewhat weaker instruction adherence | L | L | L | H | config option |
| | Local (Ollama / llama.cpp) | Private; no per-call cost | GPU or slow; weaker cite-or-abstain adherence | L* | H | M | L | |
| **Orchestration** | Plain Python | Transparent; testable; minimal dependencies | Glue code written by hand | L | L | L | M | ✅ |
| | LangChain | Integrations galore | Abstraction and API churn; opaque prompts | L | L | H | M | |
| | LangGraph | Explicit agent graphs, checkpoints | Overkill for one conditional edge | L | L | M | H | |
| **Query rewriting** | None | Predictable cost and latency | Unrecoverable vocabulary mismatch | L | L | L | H | ✅ default |
| | Conditional one-shot rewrite (on abstain) | Fixes vocabulary mismatch; happy path unaffected | ~3× cost on unanswerable queries | M | M | M | H | available |
| | Always (HyDE / multi-query) | Recall gains on some corpora | +1 LLM call on every query | H | H | M | H | |
| **Citations** | Numbered IDs + post-validation + relevance gate | Vendor-neutral; checkable; cheap | ID-level, not span-level | L | L | L | H | ✅ |
| | Native provider citations | Span-exact | Vendor lock; conflicts with JSON output | L | L | M | H | |
| | Runtime NLI verification | Strong faithfulness guarantee | Extra model; latency | M | H | H | M | |
| **Evaluation** | Custom harness (JSONL + metrics) | Deterministic retrieval half; no key needed | Small dataset; keyword proxy | L | — | L | M | ✅ |
| | RAGAS / DeepEval | Rich metric catalogue | Heavy dependencies; LLM-judged; costly | H | — | M | M | |
| **Config / deps** | pydantic-settings + uv + pyproject | Typed; validated; locked; twelve-factor | Flat namespace | L | — | L | H | ✅ |
| | YAML + Hydra | Nested configs; sweeps | Overkill; secrets separate | L | — | M | H | |
| | Poetry | Mature | Slower; no benefit over uv here | L | — | L | H | |

\* Local LLM: zero marginal cost, but high hardware cost.

---

## Narrative: why these choices for this assessment, and when the alternative wins

### Parsing — PyMuPDF

**Why here:** Page-accurate text from PDFs is the single most important parsing
property for citations, and PyMuPDF delivers it with one pip wheel on every OS. The
brief's "any document set" is dominated by text PDFs and office documents.
**When I'd choose the alternative:** Scanned archives, forms, or table-heavy financial
reports → Unstructured, or Docling with OCR. A closed-source commercial product that
can't accept AGPL → pypdf.

### Chunking — recursive, page-bounded

**Why here:** It's corpus-agnostic. With no idea what documents will arrive, we need
the method with the fewest corpus-specific parameters that still respects sentence
boundaries.
**When I'd choose the alternative:** A known corpus of well-structured Markdown/HTML
docs → header-aware chunking. Long narrative documents where topics drift within
pages, *and* evaluation shows recall problems → semantic chunking.

### Embeddings — local bge-small

**Why here:** It works offline and without a key, keeps documents on the machine, and
costs nothing per ingest. Being about 130 MB, it is small enough for an evaluator's
first run. Paired with BM25, most of its quality gap on exact tokens is covered.
**When I'd choose the alternative:** Retrieval evaluation shows recall@k is the
bottleneck *after* chunking is tuned → bge-base or large, or a hosted embedding API,
if data residency allows. A multilingual corpus → a multilingual model (config only).

### Vector store — NumPy

**Why here:** For a single-process assessment with limited infrastructure
requirements, exact in-memory search minimises operational complexity. There's no
server to start, and results are exact. At 10⁵ chunks it is single-digit
milliseconds.
**When I'd choose the alternative:** Multi-user access control, transactional
metadata or several app replicas → pgvector. Millions of vectors on one node → FAISS
(HNSW/IVF). Rich metadata filtering without running Postgres → Chroma or Qdrant.

### Retrieval — hybrid RRF

**Why here:** An unknown corpus means unknown query types. Hybrid is the
"no-regrets" default that stays robust to both paraphrase and exact-identifier
queries, and RRF needs no per-corpus weight tuning. The mode is configurable, and
evaluation reports all three.
**When I'd choose the alternative:** Latency-critical systems at huge scale where the
BM25 index is expensive → dense only, with a stronger model. Code search or log
search dominated by identifiers → BM25-heavy.

### Reranking — off by default

**Why here:** Adding it without evidence is exactly the "sophistication for its own
sake" we want to avoid. It is implemented so the hypothesis is testable with one flag.
**When I'd choose the alternative:** Evaluation shows high recall@20 but low MRR or
hit@3, meaning the right chunk is retrieved but ranked too low. Then the cross-encoder
is the targeted fix.

### LLM — Claude, hosted

**Why here:** Grounded QA quality is dominated by instruction adherence: cite, don't
invent, and abstain when unsure. A frontier hosted model at low effort gives the best
adherence, with small per-query cost on about 1.5k context tokens.
**When I'd choose the alternative:** High query volume with evaluation confirming
parity → a cheaper tier via `RAG_LLM_MODEL`. Strict data-residency rules forbidding
any external API → a local model behind `LLMProvider`, accepting weaker adherence
and a hardware cost.

### Orchestration — plain Python

**Why here:** One linear pipeline plus one conditional retry. A framework would hide
the very decisions the assessment wants to see, and would add the largest dependency
in the project.
**When I'd choose the alternative:** Multi-step agents with tool selection,
persistence of intermediate state, or human-in-the-loop checkpoints → LangGraph.

### Query rewriting — conditional, off by default

**Why here:** It targets one named failure (vocabulary mismatch → false abstention).
It adds zero cost on answerable queries, and it is bounded to one retry. It is off
until the baseline's false-abstention rate is measured.
**When I'd choose the alternative:** Users who habitually write short, keyword-poor
queries, *and* evaluation shows multi-query improves recall → always-on multi-query.
Complex multi-hop questions → query decomposition in a LangGraph agent.

### Citations — IDs + validation + gate

**Why here:** It is vendor-neutral, cheap, and produces a checkable claim-to-source
link. The gate prevents paying for an LLM call on clearly off-topic questions.
**When I'd choose the alternative:** Regulated domains (legal, medical) where
span-level provenance matters → native provider citations or runtime entailment
checks, accepting vendor lock or extra latency.

### Evaluation — custom harness

**Why here:** The retrieval half is deterministic and key-free, so architectural
comparisons (mode, chunking, reranker) can be run by anyone.
**When I'd choose the alternative:** A mature product with large evaluation sets and
budget for LLM judging → RAGAS or DeepEval for a broader metric catalogue.

---

## Cross-cutting trade-offs we explicitly accept

1. **Simplicity over scale:** in-memory store, single process. The upgrade path is
   an interface swap, not a rewrite.
2. **Privacy of documents over peak embedding quality:** local embeddings. Only the
   top-k passages leave the machine, and only when an LLM is configured.
3. **Measured over assumed sophistication:** reranking and rewriting ship disabled
   and are enabled by evidence.
4. **ID-level over span-level grounding at runtime:** stronger verification is done
   in evaluation, not on every request.
5. **Characters over tokens for chunk sizing:** no tokenizer dependency, at the cost
   of imprecision on non-English text.
