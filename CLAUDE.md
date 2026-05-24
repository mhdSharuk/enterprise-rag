# CLAUDE.md

This file provides guidance for AI coding assistants working in this repository.

---

## Project Overview

EnterpriseRAG is a hybrid retrieval-augmented generation system. It ingests enterprise JSON documents from 9 source types into Pinecone, then answers queries through a pipeline of: hybrid embedding → semantic cache → multi-source retrieval → cross-encoder reranking → chunk merging → LLM generation.

The stack uses ONNX-optimized models at inference time (no PyTorch at query time), Groq as the LLM provider via an OpenAI-compatible SDK, Pinecone for both the main vector store and the semantic cache, and Langfuse for prompt management and observability.

---

## Repository Layout

```
src/
  api.py              # FastAPI entrypoint — lifespan loads pipeline once on startup
  ui.py               # Streamlit frontend — calls the FastAPI backend at localhost:8000
  ingestion/          # Offline pipeline: JSON files → Pinecone
  retrieval/          # Online pipeline: query embedding, Pinecone fetch, rerank, merge
  cache/              # Semantic cache backed by a separate Pinecone namespace
  generation/         # LLM call via Groq; prompt fetched from Langfuse at runtime
  evaluate/           # DeepEval-based offline and online metrics
  utils/              # Logger, ONNX device detection
```

---

## Development Commands

```bash
# Install all dependencies
pip install -r requirements.txt

# Run the API server (initializes pipeline on startup)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Run the Streamlit UI (requires API running)
streamlit run src/ui.py

# Run ingestion (requires ./data/ directory with source-structured JSONs)
python -m src.ingestion.main

# Run retrieval evaluation
python -m src.evaluate.evaluate_retrieval

# Run the full pipeline manually (see bottom of src/generation/main.py for example queries)
python -m src.generation.main
```

---

## Architecture Decisions to Be Aware Of

### Dual Pinecone namespaces
The Pinecone index uses two namespaces: one for the main document store (`PINECONE_NAMESPACE`) and one for the semantic cache (`PINECONE_CACHE_NAMESPACE`). These are configured separately in the `.env` file. Do not mix them.

### ONNX at inference, PyTorch at ingestion
- At **ingestion** time, `src/ingestion/embedder.py` uses `SentenceTransformer` (PyTorch).
- At **retrieval** time, `src/retrieval/embedder.py` uses `ORTModelForFeatureExtraction` and a raw `ort.InferenceSession` for SPLADE — no PyTorch dependency at query time.
- When modifying embedders, be careful not to conflate the two separate `embedder.py` files.

### Hybrid scoring
`hybrid_score_norm()` in `src/retrieval/embedder.py` scales dense and sparse vectors by `alpha` and `(1 - alpha)` before passing to Pinecone. The `HYBRID_ALPHA` default is `0.5`. Changing this affects both cache lookups and document retrieval.

### Async Pinecone queries
`query_all_sources()` in `src/retrieval/vector_store.py` fires async Pinecone requests for all 9 sources in parallel using `async_req=True`. Results are collected by calling `.get()` on each future sequentially. Do not convert these to sequential blocking calls — it will significantly increase latency.

### `initialize_search_pipeline()` is called once
In `src/api.py`, the pipeline tuple is stored as a module-level global and initialized in the FastAPI `lifespan` context manager. All models and connections are shared across requests. Do not reinitialize per-request.

### LLM client
The Groq API is accessed via `openai.OpenAI` with a custom `base_url`. The client is instantiated once in `src/generation/config.py`. The generation model supports `reasoning_effort` as an extra body parameter.

### Prompt management
`GENERATION_SYSTEM_PROMPT` is fetched from Langfuse at import time in `src/generation/generator.py`. If Langfuse is unavailable, the module will fail to import. During development without Langfuse, replace the Langfuse fetch with a local string.

---

## Environment Variables

All configuration is via `.env` in the project root, loaded with `python-dotenv`. Each submodule loads its own config from a dedicated `config.py` — there is no single shared config module. If you add a new env variable, add it to the relevant `config.py` and document it in the README.

Key variables:

| Variable | Used in | Purpose |
|---|---|---|
| `PINECONE_API_KEY` | all modules | Pinecone auth |
| `PINECONE_INDEX_NAME` | all modules | Main + cache index name |
| `PINECONE_NAMESPACE` | retrieval, ingestion | Document namespace |
| `PINECONE_CACHE_NAMESPACE` | cache | Cache namespace |
| `DENSE_EMBEDDING_MODEL` | retrieval | HuggingFace model ID |
| `DENSE_EMBEDDING_ONNX_FILE` | retrieval | ONNX filename inside the model folder |
| `SPARSE_EMBEDDING_MODEL` | retrieval | SPLADE model ID |
| `SPARSE_EMBEDDING_ONNX_FILE` | retrieval | SPLADE ONNX filename |
| `RERANKING_MODEL` | retrieval, cache | Cross-encoder model ID |
| `RERANKING_ONNX_FILE` | retrieval, cache | Reranker ONNX filename |
| `GROQ_API_KEY` | generation, evaluate | Groq API key |
| `GROQ_BASE_URL` | generation, evaluate | Groq-compatible base URL |
| `GROQ_GENERATION_MODEL` | generation | Model string for generation |
| `EVALUATION_MODEL` | evaluate | Model string for DeepEval judge |
| `LANGFUSE_PUBLIC_KEY` | generation, evaluate | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | generation, evaluate | Langfuse secret key |
| `LANGFUSE_HOST` | generation, evaluate | Langfuse instance URL |
| `HF_TOKEN` | retrieval, ingestion | HuggingFace token for gated models |
| `LOG_LEVEL` | utils/logger | `INFO` (default) or `DEBUG` |

---

## Model Store

Downloaded and ONNX-exported models are cached under `./model_store/` with the naming convention `<HuggingFace_model_id_with_/_replaced_by_underscore>/`. On restart, models load from disk if the directory exists, skipping re-download and re-export. The `model_store/` directory is gitignored.

---

## Data Ingestion

Input data lives under `./data/` (gitignored), structured as:
```
data/
  confluence/   *.json
  github/       *.json
  jira/         *.json
  ... (9 sources total)
```

Each JSON file represents one document. The `extract_keys_from_schema()` function reads a `title_field_name` and `content_field_names` from the JSON to determine what text to embed. If these keys are absent, `FALLBACK_KEYS` is used. Chunks are assigned IDs in the format `{source}_{stem}_chunk_{index:04d}`.

---

## Evaluation Dataset

Evaluation questions live in `src/evaluate/dataset/questions.jsonl`. Each line is a JSON object with:
```json
{
  "question_id": "...",
  "question": "...",
  "expected_doc_ids": ["..."],
  "gold_answer": "...",
  "answer_facts": ["..."]
}
```

Results are written to `src/evaluate/results/`. Both the dataset and results files are gitignored.

---

## Common Pitfalls

- **Two `embedder.py` files**: `src/ingestion/embedder.py` (PyTorch/SentenceTransformer) and `src/retrieval/embedder.py` (ONNX). They are not interchangeable.
- **`reranker` is a tuple**: `load_local_reranker()` returns `(tokenizer, model)`. It is unpacked as `reranker[0]` and `reranker[1]` throughout the codebase.
- **`hdense` is a list, not a numpy array**: After `hybrid_score_norm()`, the dense vector is a plain Python list (Pinecone SDK requirement).
- **Cache is currently write-disabled**: `store_in_cache()` is commented out in `src/generation/main.py`. Re-enable it when cache population is desired.
- **`dataset_doc_uuid` metadata field required**: `query_all_sources()` reads `m["metadata"]["dataset_doc_uuid"]` as `doc_id`. Ensure this field is present in ingested vectors or retrieval will fail with a `KeyError`.
- **Streamlit chat message CSS**: Native `[data-testid="stChatMessage"]` containers are hidden via CSS in `ui.py`; all messages are rendered as raw HTML via `st.markdown(..., unsafe_allow_html=True)`.