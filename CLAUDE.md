# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an EnterpriseRAG project - a RAG (Retrieval-Augmented Generation) benchmark dataset from EnterpriseRAG-Bench. The codebase processes JSON documents from 9 enterprise sources and provides a multi-stage retrieval system for AI applications.

## Architecture

The project consists of three main services:

### 1. Ingestion Service (`src/ingestion/`)
- Processes JSON files from 9 enterprise sources
- Converts to Markdown chunks
- Ingests dense + sparse embeddings to Pinecone
- Located in: `src/ingestion/` and `misc/process_all_json.ipynb`

### 2. Retrieval Service (Planned)
- Two-stage cross-encoder reranking
- Per-source search with metadata filtering
- Cross-source final reranking

### 3. Generation Service (Planned)
- For final answer generation

---

## Ingestion Pipeline

### Current Implementation

The ingestion pipeline runs in Google Colab via `misc/process_all_json.ipynb`:

1. **Install dependencies**: langchain-text-splitters, pinecone-client, sentence-transformers
2. **Load models**:
   - Qwen2.5-Coder-1.5B-Instruct for key extraction
   - BAAI/bge-large-en-v1.5 for dense embeddings (1024-dim)
   - Pinecone inference API for sparse embeddings
3. **Process each JSON file**:
   - Extract embedding_keys vs metadata_keys using Qwen
   - Convert JSON → Markdown using Qwen
   - Split into chunks using langchain splitters
4. **Upsert to Pinecone**:
   - Dense: BGE-large embeddings
   - Sparse: Pinecone's `pinecone-sparse-english-v0` model
   - Metadata: source, chunk_index, all metadata_keys from Qwen

### Pinecone Vector Format
```python
{
    "id": "confluence_filename_chunk_0001",
    "values": dense_embedding_1024d,
    "sparse_values": {
        "indices": [array of integers],
        "values": [array of floats]
    },
    "metadata": {
        "source": "confluence",
        "source_path": "data/confluence/...",
        "file_name": "...",
        "chunk_index": 0,
        "text": "full chunk text...",
        # All metadata_keys from Qwen extraction
        "author": "...",
        "created_at": "...",
        "labels": [...],
        ...
    }
}
```

### Metadata Filtering
- Uses metadata-based filtering (not namespaces)
- Filter by source: `{"source": {"$eq": "confluence"}}`
- Supports any field in metadata for filtering

---

## Code Structure

### Key Files

| File | Purpose |
|------|---------|
| `misc/process_all_json.ipynb` | Ingestion notebook (runs in Colab) |
| `src/ingestion/` | Modular Python package |
| `.env` | Configuration (NOT committed) |

### Ingestion Notebook Cells

| Cell | Content |
|------|---------|
| 1 | Install dependencies |
| 2 | Imports |
| 3 | Hardcoded config (for Colab) |
| 4 | Initialize Pinecone client |
| 5 | Load Qwen model |
| 6 | Load BGE embedding model |
| 7 | Prompts |
| 8 | extract_keys_with_llm() |
| 9 | convert_to_markdown() |
| 10 | split_markdown() |
| 11 | process_file_to_pinecone() - main ingestion logic |
| 12 | process_all_files() |
| 13 | Run |

---

## Data Sources

The `data/` folder contains 175 JSON files from 9 enterprise sources:
- `confluence/` - Company documentation
- `fireflies/` - Meeting transcripts
- `github/` - Pull request descriptions
- `gmail/` - Email threads
- `google_drive/` - Shared drive documents
- `hubspot/` - CRM company data
- `jira/` - Support tickets
- `linear/` - Engineering tasks
- `slack/` - Channel messages

---

## Planned Features

### Two-Stage Retrieval

The retrieval architecture (planned):
1. **Parallel metadata-filtered search** - Query each source namespace/filter in parallel, get top 50 per source
2. **Stage 1 reranking** - Cross-encoder rerank within each source (top 50 → 10)
3. **Stage 2 reranking** - Cross-encoder across all sources (total → top 10)

### Recommended Models
- Dense: BAAI/bge-large-en-v1.5 (already in use)
- Stage 1/2 Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

### Hybrid Search Query (For Implementation in Next Phase)

The query must balance two signals using an "Alpha" weight:

1. **Alpha Weighting**: Manually scale query vectors before calling `.query()`:
   - Scaled_Dense = Dense_Query * alpha
   - Scaled_Sparse = Sparse_Query * (1 - alpha)

2. **Alpha Tuning**:
   - `alpha=1.0` for 100% Semantic (Dense)
   - `alpha=0.0` for 100% Keyword (Sparse)
   - `alpha=0.5` for balanced hybrid search

---

## Running the Pipeline

### Ingestion (Google Colab)

1. Copy `misc/process_all_json.ipynb` to Google Colab
2. Replace `PINECONE_API_KEY = "your_key_here"` with actual key
3. Run cells in order (cell-1 to cell-13)

### Testing Retrieval

Query Pinecone with:
```python
# Dense + sparse hybrid search
index.query(
    vector=dense_embedding,
    sparse_vector={"indices": [...], "values": [...]},
    filter={"source": {"$eq": "confluence"}},
    top_k=50
)
```

---

## Configuration

All configuration is hardcoded in notebooks for Colab compatibility:
- Pinecone API key
- Index name
- Model IDs
- Chunk sizes

NOTEs:
- `.env` files are gitignored
- No CLI arguments - use hardcoded values for Colab
- Use simple module-level variables, avoid fancy caching patterns
- Keep imports absolute: `from src.ingestion.config import MODEL_ID`