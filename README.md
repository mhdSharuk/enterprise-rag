# EnterpriseRAG

A production-grade Retrieval-Augmented Generation (RAG) system for querying enterprise knowledge across multiple data sources. It combines hybrid search (dense + sparse embeddings), semantic caching, cross-encoder reranking, and an LLM generation step — all served via a FastAPI backend with a Streamlit UI.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         FastAPI Backend         │  ← src/api.py
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│        Semantic Cache Check     │  ← src/cache/
│  (Pinecone hybrid + reranker)   │
└─────────────────────────────────┘
    │ Cache Miss
    ▼
┌─────────────────────────────────┐
│     Hybrid Retrieval            │  ← src/retrieval/
│  Dense (ONNX) + Sparse (SPLADE) │
│  → Pinecone query per source    │
│  → Cross-encoder reranking      │
│  → Chunk merging                │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│        LLM Generation           │  ← src/generation/
│    (Groq API via OpenAI SDK)    │
└─────────────────────────────────┘
    │
    ▼
  Answer + Metadata
```

---

## Project Structure

```
.
├── src/
│   ├── api.py                  # FastAPI app with /query, /health endpoints
│   ├── ui.py                   # Streamlit chat frontend
│   │
│   ├── ingestion/              # Data ingestion pipeline
│   │   ├── main.py             # Orchestrates file processing & upsertion
│   │   ├── embedder.py         # Dense + sparse embedding for ingestion
│   │   ├── text_processor.py   # JSON → text conversion + chunking
│   │   ├── key_extractor.py    # LLM-based schema key extraction
│   │   ├── vector_store.py     # Pinecone index creation & upsert
│   │   ├── prompts.py          # Langfuse-managed prompts
│   │   └── config.py
│   │
│   ├── retrieval/              # Query-time retrieval
│   │   ├── query.py            # Retrieval orchestration
│   │   ├── embedder.py         # ONNX dense + SPLADE sparse embedders
│   │   ├── reranker.py         # ONNX cross-encoder reranker
│   │   ├── vector_store.py     # Pinecone async multi-source querying
│   │   ├── chunk_utils.py      # Adjacent chunk merging logic
│   │   └── config.py
│   │
│   ├── cache/                  # Semantic query cache
│   │   ├── main.py             # Cache check & store logic
│   │   └── config.py
│   │
│   ├── generation/             # LLM response generation
│   │   ├── main.py             # Full pipeline: embed → cache → retrieve → generate
│   │   ├── generator.py        # Groq/OpenAI-compatible generation call
│   │   └── config.py
│   │
│   ├── evaluate/               # Evaluation framework
│   │   ├── evaluate_retrieval.py   # Recall@K, MRR, contextual recall
│   │   ├── llm_metrics.py          # Faithfulness, answer correctness, relevancy
│   │   ├── utils.py                # Batch embedding & reranking helpers
│   │   └── config.py
│   │
│   └── utils/
│       ├── logger.py           # Centralized logging
│       └── check_device.py     # CUDA / CPU ONNX provider detection
│
├── requirements.txt
└── .gitignore
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> For GPU support with ONNX Runtime:
> ```bash
> pip install onnxruntime-gpu
> ```

> For llama-cpp-python (optional local inference):
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --no-cache-dir --force-reinstall
> ```

### 2. Configure environment

Create a `.env` file in the project root:

```env
# Pinecone
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_NAMESPACE=
PINECONE_CACHE_NAMESPACE=
PINECONE_DIMENSION=
PINECONE_METRIC=
PINECONE_CLOUD=
PINECONE_REGION=

# Models
DENSE_EMBEDDING_MODEL=        # e.g. BAAI/bge-large-en-v1.5
DENSE_EMBEDDING_ONNX_FILE=    # e.g. model_optimized.onnx
SPARSE_EMBEDDING_MODEL=       # e.g. naver/splade-cocondenser-ensemble-distil
SPARSE_EMBEDDING_ONNX_FILE=   # e.g. model.onnx
RERANKING_MODEL=              # e.g. cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKING_ONNX_FILE=          # e.g. model.onnx

# Groq (LLM generation)
GROQ_API_KEY=
GROQ_BASE_URL=
GROQ_GENERATION_MODEL=
EVALUATION_MODEL=

# Langfuse (observability & prompt management)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=

# HuggingFace
HF_TOKEN=

# Logging
LOG_LEVEL=INFO
```

---

## Running the System

### Start the API server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will initialize the full pipeline on startup (model loading may take a minute).

### Start the Streamlit UI

```bash
streamlit run src/ui.py
```

Ensure the API server is running before starting the UI — the frontend queries `http://localhost:8000`.

---

## API Reference

### `GET /health`
Returns pipeline status.

```json
{ "status": "healthy", "pipeline_loaded": true }
```

### `POST /query`
Submit a question to the RAG system.

**Request:**
```json
{ "question": "What are the SLA commitments for the healthcare client?" }
```

**Response:**
```json
{
  "answer": "...",
  "is_cached": false,
  "tokens_used": 842,
  "time_taken": 3.21
}
```

Swagger UI available at `http://localhost:8000/docs`.

---

## Data Ingestion

Place your JSON documents under a `./data/` directory structured by source (e.g. `data/confluence/`, `data/github/`, etc.).

Run the ingestion pipeline:

```bash
python -m src.ingestion.main
```

Each JSON file is processed as follows:

1. Schema keys are extracted to determine what gets embedded vs. stored as metadata.
2. Content is converted to plain text and split into overlapping chunks (default: 1200 chars, 200 overlap).
3. Each chunk is embedded with both dense and sparse models and upserted into Pinecone with source-tagged metadata.

Supported sources: `confluence`, `fireflies`, `github`, `gmail`, `google_drive`, `hubspot`, `jira`, `linear`, `slack`.

---

## Retrieval Pipeline

At query time:

1. **Embedding** — the query is embedded with both the dense (ONNX) model and the SPLADE sparse model. Hybrid scores are computed with a configurable alpha (default `0.5`).
2. **Cache lookup** — a hybrid search against the cache namespace checks for a semantically similar past query. Results above the reranker threshold (`0.9`) are returned immediately.
3. **Multi-source retrieval** — async Pinecone queries are issued in parallel across all 9 sources (top-5 per source by default).
4. **Reranking** — a cross-encoder reranker scores all retrieved chunks against the query and selects the top-N.
5. **Chunk merging** — adjacent retrieved chunks from the same document are merged to preserve context.
6. **Generation** — the merged context is passed to the Groq LLM with a Langfuse-managed system prompt.

---

## Evaluation

### Retrieval evaluation

```bash
python -m src.evaluate.evaluate_retrieval
```

Computes Recall@5, Recall@10, MRR, and LLM-based contextual recall (via DeepEval + Groq). Scores are logged to Langfuse.

### Generation / LLM evaluation

Metrics available in `src/evaluate/llm_metrics.py`:

- Contextual Recall
- Contextual Relevancy
- Answer Relevancy
- Faithfulness
- Answer Correctness (GEval)

Both offline (against a golden dataset) and online (live query evaluation) modes are supported.

---

## Key Configuration Parameters

| Parameter | Location | Default | Description |
|---|---|---|---|
| `TOP_K_PER_SOURCE` | `retrieval/config.py` | `5` | Chunks retrieved per source |
| `RERANK_TOP_N` | `retrieval/config.py` | `10` | Final chunks kept after reranking |
| `HYBRID_ALPHA` | `retrieval/config.py` | `0.5` | Dense vs. sparse weight (1 = dense only) |
| `RERANK_THRESHOLD` | `retrieval/config.py` | `0.5` | Minimum reranker score to include a chunk |
| `CACHE_SCORE_THRESHOLD` | `cache/config.py` | `0.9` | Reranker score threshold for cache hit |
| `CACHE_SEMANTIC_TOP_K` | `cache/config.py` | `5` | Candidates fetched from cache namespace |
| `CHUNK_SIZE` | `ingestion/config.py` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `ingestion/config.py` | `200` | Overlap between adjacent chunks |

---

## Model Storage

Downloaded and exported ONNX models are cached under `./model_store/` using the pattern `<model_name_with_slashes_replaced_by_underscores>/`. On subsequent startups, models are loaded from disk rather than re-downloaded.

---

## Observability

All prompts are managed in **Langfuse** and fetched at runtime by label (`production`). Retrieval and generation evaluation scores are logged as Langfuse spans. Set `LOG_LEVEL=DEBUG` in `.env` for verbose output.