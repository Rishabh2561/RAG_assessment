# 06 — Evaluation

## 1. Principles

- **Separate the stages.** A wrong answer can come from retrieval, ranking,
  generation or citation, and each has a different fix. We measure each stage on its
  own.
- **The retrieval half needs no LLM**, so it is deterministic, free, and runnable by
  anyone (`rag eval`).
- **Evaluation drives defaults.** Three defaults in this repository were changed by
  measurement: retrieval mode, chunk size and relevance threshold. See
  [DECISION_LOG.md](DECISION_LOG.md) D8–D11.

## 2. Dataset structure

The dataset is JSONL, one item per line
([`data/eval/northwind_eval.jsonl`](../data/eval/northwind_eval.jsonl)):

```json
{"id": "q13", "category": "exact_id", "question": "What does E-221 mean?",
 "answerable": true,
 "expected_sources": ["nw200_operator_manual.pdf"],
 "expected_evidence": ["E-221 Battery overheating"],
 "expected_keywords": ["battery", "overheating"]}
```

| Field | Use |
|-------|-----|
| `expected_sources` | Document-level relevance, and source recall. |
| `expected_evidence` | **Chunk-level** relevance: a chunk is relevant if it comes from an expected source *and* contains an evidence string. Labelling by text, not chunk ID, keeps labels valid when chunk size changes, which is what makes the chunk-size sweep possible. |
| `expected_keywords` | A cheap proxy for answer correctness (keyword recall). |
| `answerable` | Ground truth for abstention metrics, and for gate calibration. |
| `category` | Slices metrics by failure type. |

The sample set has 26 questions over 6 documents (4 formats), in six categories:

| Category | n | Purpose |
|----------|---|---------|
| factual | 6 | Wording close to the source (baseline). |
| paraphrase | 8 | Different vocabulary from the source: probes the dense model's semantic matching. |
| exact_id | 4 | Error codes and model numbers: probes lexical matching. |
| conflict | 1 | The manual says 8 h of battery; the firmware notes say 9.5 h. The answer must surface both. |
| multi_part | 2 | Several facts in one answer. |
| unanswerable | 5 | Four are on-topic ("CEO?", "list price?", "pension?", "dogs?") and one is off-topic ("sourdough"). |

A test (`tests/integration/test_evaluation.py`) asserts that every answerable label
is satisfiable by at least one chunk of the corpus, so the labels can't silently rot.

## 3. Metrics

### 3.1 Retrieval quality (no LLM)

| Metric | Definition | Question it answers |
|--------|------------|---------------------|
| hit@k (k = 1, 3, 5, 10) | Share of answerable questions with ≥ 1 relevant chunk in the top k | "Does the context window contain the evidence?" hit@`top_k` is the key number. |
| MRR@10 | Mean of 1 / rank of the first relevant chunk | "How highly is the evidence ranked?" |
| source recall@5 | Share of expected documents represented in the top 5 | Multi-document and conflict questions. |
| hit@k by category | The above, sliced by category | "Which query type fails?" |
| latency p50 / p95 | Retrieval (+ rerank) wall time | Cost of each retrieval option. |

### 3.2 Answer quality (LLM required, `--answers`)

| Metric | Definition |
|--------|------------|
| keyword recall | Share of `expected_keywords` present in answers to answerable questions. |
| abstention accuracy | Share of questions where predicted `answerable` equals the label. |
| false abstention rate | Share of answerable questions where the system said "not found". |
| false answer rate | Share of unanswerable questions that got an answer (the hallucination risk). |

### 3.3 Grounding and faithfulness

| Metric | Definition |
|--------|------------|
| grounded rate | Share of answered questions with status `grounded` (≥ 1 valid citation). |
| faithfulness (`--judge`) | An LLM judge checks that every claim in the answer is supported by the *cited* passages, giving a 0/1 score per answer. |

### 3.4 Citation correctness

| Metric | Definition |
|--------|------------|
| citation validity | Share of answers with no citation to a source ID that wasn't supplied. |
| citation hit rate | Share of answered, answerable questions where ≥ 1 cited chunk is a *relevant* chunk (per the evidence labels). |

### 3.5 Latency and cost

End-to-end latency p50 / p95 comes from `QueryTrace.total_ms`, with per-stage
timings in the trace. Average input and output tokens per question are recorded as
well. Cost per question = tokens × the model's price (Opus 5.5: $4 / $20 per MTok in
and out). At about 1.5k input and 150 output tokens, that is roughly **$0.009 per
question**, or about 3× that for unanswerable questions when rewriting is enabled.

### 3.6 Failure cases

The per-item rows in the JSON report (`eval_reports/*.json`) list, for each question,
the first relevant rank, the best dense score, status, citations, attempts and
tokens. Failures are diagnosed from these rows (see §5).

## 4. Results

### 4.1 Retrieval sweep (measured)

This comes from `python scripts/retrieval_sweep.py --rerank`: bge-small-en-v1.5, 6
documents, 26 questions (21 answerable), CPU only.

| chunk size | chunks | mode | reranker | hit@1 | hit@3 | hit@5 | MRR@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| 400 | 49 | dense | none | 0.905 | 0.952 | 1.000 | 0.940 | 51 |
| 400 | 49 | bm25 | none | 0.810 | 0.857 | 0.857 | 0.825 | 0.1 |
| 400 | 49 | hybrid | none | 0.857 | 0.905 | 0.905 | 0.886 | 51 |
| 400 | 49 | hybrid | cross_encoder | 0.905 | 0.952 | 0.952 | 0.935 | 1486 |
| **700** | **28** | **dense** | **none** | **0.905** | **0.952** | **1.000** | **0.933** | **12** |
| 700 | 28 | bm25 | none | 0.714 | 0.857 | 0.857 | 0.770 | 0.1 |
| 700 | 28 | hybrid | none | 0.810 | 0.905 | 0.905 | 0.854 | 15 |
| 700 | 28 | dense | cross_encoder | 0.905 | 0.952 | 0.952 | 0.934 | 1490 |
| 1000 | 17 | dense | none | 0.810 | 0.952 | 1.000 | 0.877 | 11 |
| 1000 | 17 | hybrid | none | 0.810 | 0.905 | 0.952 | 0.874 | 10 |
| 1000 | 17 | dense | cross_encoder | 1.000 | 1.000 | 1.000 | 1.000 | 1758 |

(Selected rows; the full output is written to `eval_reports/retrieval_sweep.json`.)
The **bold** row is the shipped default.

hit@3 by category (chunk size 700, no reranker):

| mode | factual | paraphrase | exact_id | multi_part | conflict |
|------|---------|------------|----------|------------|----------|
| dense | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 |
| bm25 | 1.00 | 0.62 | 1.00 | 1.00 | 1.00 |
| hybrid | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 |

**Reading the results:**

1. **BM25 fails on paraphrase, as expected**, and doesn't beat dense on exact IDs.
   On this corpus the dense model already resolves "E-221", "E-310" and "NW-C2".
2. **Hybrid is worse than dense here.** The diagnosis for q10 ("How long does it take
   to fully recharge the warehouse robot?"): dense ranks the evidence 4th. BM25
   doesn't retrieve it at all, because there's no stemming and "recharge" ≠
   "charging". RRF then ranks chunks found by *both* lists above it, and it falls out
   of the top 10. This is a structural weakness of equal-weight RRF when one
   retriever is blind to a query, and it's why hybrid is not the default (D8).
3. **Chunk size:** 400 and 700 rank better than 1000 (hit@1 0.905 vs 0.810). We chose
   700 because it matches 400's quality with 43% fewer chunks and passages that hold
   more self-contained context (D10).
4. **The reranker** only helped at chunk size 1000, where it raised hit@1 to 1.00.
   Because dense hit@5 is already 1.00 and the LLM reads the top 5, the reranker
   reorders evidence that's already in context, at about 1.5 s per query. It stays
   off (D11).
5. **Caveat:** 21 answerable questions give directional evidence, not statistical
   significance. The differences between dense and hybrid are one to two questions.

### 4.2 Relevance-gate calibration (measured)

`rag eval` reports the distribution of the best dense score for answerable and
unanswerable questions, and the accuracy-maximising threshold.

| | min | max |
|---|---|---|
| answerable (21) | 0.593 | 0.860 |
| on-topic unanswerable (4) | 0.630 | 0.735 |
| off-topic unanswerable (1) | 0.472 | 0.472 |

The overlap is intrinsic. "What is the list price of an NW-200?" is semantically
close to the manual even though the manual doesn't contain the answer. **A similarity
threshold can't detect on-topic unanswerable questions.** Only the generator can, by
reading the passages. So the gate is set at 0.50, below every answerable score, and
filters only clearly off-topic questions (D9). `tests/e2e/test_real_models.py`
asserts this invariant.

### 4.3 Answer quality: not yet measured

No LLM API key was available during development, so the generation metrics (§3.2–3.4)
**have not been run against Claude**. The harness is implemented and tested with a
scripted fake LLM (`tests/integration/test_evaluation.py`). To produce the numbers:

```bash
export ANTHROPIC_API_KEY=...
rag ingest data/sample/northwind -c northwind
rag eval data/eval/northwind_eval.jsonl -c northwind --modes dense --answers --judge
```

The expected cost is 26 questions × (1 answer call + ≤ 1 judge call), about $0.50
with Opus 5.5.

## 5. How evaluation drives architecture

| Observation | Likely cause | What to change (in order) |
|-------------|--------------|---------------------------|
| Low hit@`top_k` (evidence not in context) | Chunking or embedding | 1) chunk size and overlap; 2) a larger embedding model (`RAG_EMBEDDING_MODEL`); 3) `hybrid` if the failures are in `exact_id`; 4) enable rewriting if the failures are in `paraphrase`. |
| High hit@20 but low hit@`top_k` | Ranking | Enable the cross-encoder (`RAG_RERANKER=cross_encoder`), or raise `RAG_TOP_K` if the context budget allows. |
| Good hit@`top_k`, low keyword recall or faithfulness | Generation | Prompt wording, a stronger model or higher effort, a larger `RAG_MAX_CONTEXT_CHARS`. |
| High false-abstention rate with good retrieval | Generation is over-cautious | Loosen the prompt's abstention wording. Check `max_context_chars` isn't truncating evidence. |
| High false-abstention rate with poor retrieval on paraphrases | Vocabulary mismatch | `RAG_QUERY_REWRITE=true` (ADR-009). Compare `attempts` and cost before and after. |
| High false-answer rate | Hallucination on unanswerable questions | Strengthen the abstention instruction. Check whether a stale gate threshold now lets off-topic questions through. |
| Invalid citations > 0 | The model cites IDs it wasn't given | Prompt; model choice. Already mitigated at runtime (invalid IDs are dropped, status `unverified`). |
| Citation hit rate low but keyword recall high | The model answers from a passage other than the evidence (or from prior knowledge) | Examine faithfulness. Tighten "only from the sources". |
| Latency p95 too high | Reranker or LLM | Disable the reranker, lower effort, or choose a smaller model (`RAG_LLM_MODEL`). |

## 6. Evaluating a new document set

1. Write 20–50 questions that reflect real usage, including at least 20%
   unanswerable questions, some on-topic and some off-topic, with evidence strings.
2. `rag ingest <docs> -c new` then `rag eval new.jsonl -c new`. Read hit@5 by
   category and the gate calibration line.
3. If the answerable minimum drops below `RAG_MIN_RELEVANCE`, lower the threshold.
   Keep a margin: false abstentions are worse than one extra LLM call.
4. Sweep chunk size if hit@5 < 0.9 (`scripts/retrieval_sweep.py` shows the pattern).
5. With a key: `--answers --judge`, and act on the table above.
