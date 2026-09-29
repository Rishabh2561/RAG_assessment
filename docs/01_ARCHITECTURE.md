# 01 — Architecture

## 1. High-level view

The system is two independent pipelines that share one persisted **collection**:

- **Ingestion** (write path): file → parse → chunk → embed → store.
- **Query** (read path): question → retrieve → (rerank) → gate → prompt → LLM →
  validate citations → answer.

Both pipelines are assembled from small components behind interfaces by a single
**factory** that reads a typed `Settings` object. The CLI and the HTTP API are thin
adapters over two services: `IngestionService` and `QueryService`. No business logic
lives in the interfaces.

```mermaid
flowchart LR
    subgraph Interfaces
        CLI["CLI (typer)"]
        API["REST API (FastAPI)"]
    end

    subgraph Orchestration
        F["factory.build_*()"]
        IS[IngestionService]
        QS[QueryService]
    end

    subgraph Ingestion
        PR[ParserRegistry] --> P1[PdfParser<br/>PyMuPDF]
        PR --> P2[TextParser<br/>txt / md]
        PR --> P3[DocxParser]
        CH[RecursiveChunker]
    end

    subgraph Knowledge["Collection (on disk)"]
        VS[(VectorStore<br/>NumpyVectorStore)]
        CAT[(DocumentCatalog)]
    end

    subgraph Retrieval
        R{Retriever}
        DR[DenseRetriever]
        LR[BM25Retriever]
        HR[HybridRetriever<br/>RRF]
        RR[Reranker<br/>none / cross-encoder]
    end

    subgraph Generation
        PB[Prompt builder]
        LLM[LLMProvider<br/>Anthropic / none]
        CV[Citation validator]
    end

    EMB[EmbeddingProvider<br/>fastembed / hashing]
    CFG[(Settings<br/>env + .env)]

    CLI --> IS & QS
    API --> IS & QS
    CFG --> F --> IS & QS
    IS --> PR --> CH --> EMB --> VS
    IS --> CAT
    QS --> R
    R --- DR & LR & HR
    DR --> EMB
    DR --> VS
    LR --> VS
    HR --> DR & LR
    R --> RR --> PB --> LLM --> CV --> QS
```

## 2. Components

| Component | Responsibility | Default implementation | Swappable via |
|-----------|----------------|------------------------|---------------|
| `Settings` | Single typed source of configuration: env vars with the `RAG_` prefix plus an optional `.env`. Validated at start-up. | `pydantic-settings` | — |
| `ParserRegistry` / `DocumentParser` | Map a file extension to a parser. Produce a `ParsedDocument` made of ordered `Section`s, each with an optional page number. Detect empty, corrupt and scanned files. | `PdfParser` (PyMuPDF), `TextParser`, `DocxParser` | Register another parser |
| `RecursiveChunker` | Split each section into overlapping chunks, preferring paragraph → line → sentence → word boundaries. Chunks never cross page boundaries, so page citations stay exact. | Size and overlap in characters | `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP` |
| `EmbeddingProvider` | Text → L2-normalised vectors. Separate `embed_documents` and `embed_query` methods (some models use different prefixes). | `FastEmbedProvider` (`BAAI/bge-small-en-v1.5`, local ONNX) | `RAG_EMBEDDING_PROVIDER`, `RAG_EMBEDDING_MODEL` |
| `VectorStore` | Persist chunks and vectors per collection; exact cosine search; delete by document. Records the embedding model and dimension, and refuses mismatched queries. | `NumpyVectorStore` (`.npy` + `.jsonl`) | `RAG_VECTOR_STORE` |
| `DocumentCatalog` | Per-collection registry of documents: ID, file name, content hash, chunk count, ingestion time. Drives idempotent re-ingestion and listing. | JSON file | — |
| `Retriever` | Question → ranked `RetrievedChunk`s. | `HybridRetriever` (dense + BM25 fused with reciprocal-rank fusion) | `RAG_RETRIEVAL_MODE` = `dense` \| `bm25` \| `hybrid` |
| `Reranker` | Re-score the top-N candidates and keep the top-k. | `NoOpReranker` | `RAG_RERANKER` = `none` \| `cross_encoder` |
| Relevance gate | Abstain *before* calling the LLM when the best dense similarity is below a threshold. | Threshold from settings | `RAG_MIN_RELEVANCE` |
| Prompt builder | Render retrieved chunks as numbered sources `[S1]…[Sn]` with file and page, inside a context budget. Documents are delimited as untrusted data. | — | `RAG_MAX_CONTEXT_CHARS` |
| `LLMProvider` | (system, user, schema) → structured JSON plus usage. | `AnthropicProvider` (Claude) | `RAG_LLM_PROVIDER` = `anthropic` \| `none` |
| Citation validator | Keep only citations whose IDs were actually supplied as context; derive `grounding_status`. | — | — |
| `QueryService` | Run the query flow, including the optional corrective retry, and build a `QueryTrace`. | — | `RAG_QUERY_REWRITE` |
| `IngestionService` | Run the ingestion flow per file, isolating failures and producing an `IngestReport`. | — | — |
| Evaluation runner | Run a labelled dataset through retrieval and, optionally, generation, and compute metrics. | `rag eval` | — |

## 3. Ingestion flow

```mermaid
sequenceDiagram
    participant U as User (CLI / API)
    participant IS as IngestionService
    participant PR as ParserRegistry
    participant CH as Chunker
    participant E as EmbeddingProvider
    participant VS as VectorStore
    participant C as DocumentCatalog

    U->>IS: ingest(paths, collection)
    loop each file (failures isolated per file)
        IS->>IS: sha256(bytes) → doc_id
        IS->>C: already have doc_id?
        alt unchanged duplicate
            IS-->>U: status=skipped_duplicate
        else new or changed
            IS->>PR: parse(path)
            PR-->>IS: ParsedDocument (sections with page numbers)
            IS->>CH: chunk(document)
            CH-->>IS: Chunk[] (deterministic ids)
            IS->>E: embed_documents(texts) in batches
            E-->>IS: vectors
            IS->>VS: delete(old doc with same source name), add(chunks, vectors)
            IS->>C: upsert(DocumentRecord)
            IS-->>U: status=ingested, chunks=N
        end
    end
    IS->>VS: persist()
    IS->>C: persist()
```

Key properties:

- **Document identity = content hash.** Re-uploading identical bytes (even under a
  different name) is a no-op. Uploading different bytes under an existing file name
  replaces the previous version, so the index never holds two versions of the same
  file.
- **Deterministic chunk IDs** (`{doc_id[:12]}-{index:05d}`), so citations, logs and
  evaluation labels stay stable across runs.
- **Per-file isolation.** A corrupt file yields `status=failed` with a reason, and the
  rest of the batch continues.
- **Persist once per batch**, after all files, using an atomic write-and-rename
  (§7).

## 4. Query flow

```mermaid
flowchart TD
    Q[Question] --> V{valid?<br/>non-empty, ≤ max length}
    V -- no --> E1[400 / ValidationError]
    V -- yes --> R[Retrieve top-N candidates<br/>dense / bm25 / hybrid]
    R --> RR[Rerank → top-k<br/>no-op by default]
    RR --> G{best dense score<br/>≥ min_relevance?}
    G -- no --> AB1[Abstain: 'not found in documents'<br/>no LLM call]
    G -- yes --> L{LLM configured?}
    L -- no --> RO[Retrieval-only result:<br/>ranked passages, no generated answer]
    L -- yes --> P[Build prompt: numbered sources S1..Sk]
    P --> GEN[LLM → answerable, answer, cited ids]
    GEN --> CV[Validate citations against supplied ids]
    CV --> A{answerable?}
    A -- yes --> OUT[Answer + citations + trace]
    A -- no --> RW{rewrite enabled<br/>and not yet retried?}
    RW -- no --> AB2[Abstain with LLM's reason]
    RW -- yes --> QR[LLM rewrites query] --> R2[Retrieve again,<br/>merge with first results] --> P
```

The corrective branch (`RAG_QUERY_REWRITE=true`) is the only agentic behaviour in
the system. It is bounded to **one** retry and is off by default. The rationale is in
ADR-009.

## 5. Runtime document lifecycle

| Stage | Trigger | Effect |
|-------|---------|--------|
| Upload / ingest | `rag ingest <paths> -c <collection>` or `POST /collections/{c}/documents` | File is parsed, chunked, embedded and persisted. Uploaded bytes go to a temporary file, which is deleted once parsing finishes (§09). |
| Re-ingest same content | Same commands | Skipped (`skipped_duplicate`). |
| Re-ingest changed file with the same name | Same commands | Old chunks for that file name are removed, and the new version is indexed. |
| List | `rag docs -c <c>` / `GET /collections/{c}/documents` | Reads the catalog. |
| Delete | `rag delete <doc_id> -c <c>` / `DELETE /collections/{c}/documents/{doc_id}` | Removes the document's chunks and vectors and its catalog entry. |
| Drop collection | `rag drop -c <c>` / `DELETE /collections/{c}` | Removes the collection directory. |
| Query | `rag ask` / `POST /collections/{c}/query` | Read-only against the current snapshot. |

A collection is created lazily on first ingest. Querying an empty or missing
collection returns a clear `CollectionEmpty` error rather than an empty answer.

## 6. Configuration flow

```mermaid
flowchart LR
    ENV[Environment variables RAG_*] --> S
    DOT[.env file] --> S
    DEF[Code defaults] --> S
    S[Settings pydantic model<br/>validated at start-up] --> F[factory]
    F -->|RAG_EMBEDDING_PROVIDER| EMB[EmbeddingProvider]
    F -->|RAG_VECTOR_STORE| VS[VectorStore]
    F -->|RAG_RETRIEVAL_MODE| RET[Retriever]
    F -->|RAG_RERANKER| RR[Reranker]
    F -->|RAG_LLM_PROVIDER| LLM[LLMProvider]
    CLIF[CLI flags e.g. --top-k, --mode] -.override.-> S
```

Precedence is CLI flag > environment variable > `.env` > default. Invalid values
(for example `chunk_overlap >= chunk_size`, or an unknown provider) fail at start-up
with a message naming the variable. The factory is the **only** module that imports
concrete vendor classes. Everything else depends on the protocols in each package's
`base.py`.

## 7. Persistence layout

```
${RAG_DATA_DIR}/                 (default ./.rag_data)
└── collections/
    └── <collection>/
        ├── manifest.json        embedding model, dimension, schema version, counts
        ├── chunks.jsonl         one Chunk per line (text + metadata)
        ├── vectors.npy          float32 [n_chunks, dim], row-aligned with chunks.jsonl
        └── documents.json       DocumentCatalog
```

Writes go to `*.tmp` and are then renamed with `os.replace`, which is atomic on the
same filesystem, so a crash mid-write leaves the previous consistent snapshot in
place. The BM25 index is **derived** state: it is rebuilt in memory from
`chunks.jsonl` when a collection loads, and never persisted.

## 8. Failure paths (summary)

Full detail is in [08_FAILURE_MODES.md](08_FAILURE_MODES.md).

| Where | Failure | Behaviour |
|-------|---------|-----------|
| Start-up | Invalid config | Exit with a message naming the bad variable. |
| Ingest | Unsupported, empty, corrupt or scanned file | That file is reported as `failed` / `skipped` with a reason; the batch continues. |
| Ingest | Embedding model fails to load or download | Typed `EmbeddingError`; nothing is persisted for the batch. |
| Query | Collection missing or empty | `CollectionNotFound` / `CollectionEmpty` error (HTTP 404 / 409). |
| Query | Embedding model differs from the one the index was built with | `IndexMismatchError` telling you to re-ingest. |
| Query | Low relevance | Abstain without calling the LLM. |
| Query | LLM timeout, rate limit or 5xx | SDK retries with backoff (bounded); then `GenerationError` (HTTP 503). The trace records the failure. |
| Query | LLM refusal or malformed output | `GenerationError` with the reason; never shown to the user as an answer. |
| Query | LLM cites IDs it wasn't given | Invalid citations are dropped. If none remain, `grounding_status = "unverified"` and a warning is logged. |
