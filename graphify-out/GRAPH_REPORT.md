# Graph Report - D:\Projects\enterprise-rag\src  (2026-06-06)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 104 nodes · 125 edges · 19 communities (15 shown, 4 thin omitted)
- Extraction: 82% EXTRACTED · 18% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bc3794a9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 10 edges
2. `run_query()` - 8 edges
3. `process_file()` - 8 edges
4. `query_endpoint()` - 5 edges
5. `initialize_search_pipeline()` - 5 edges
6. `get_embeddings()` - 5 edges
7. `RerankResult` - 5 edges
8. `get_onnx_provider()` - 5 edges
9. `QueryRequest` - 4 edges
10. `QueryResponse` - 4 edges

## Surprising Connections (you probably didn't know these)
- `lifespan()` --calls--> `initialize_search_pipeline()`  [INFERRED]
  api.py → generation/main.py
- `query_endpoint()` --calls--> `run_query()`  [INFERRED]
  api.py → generation/main.py
- `run_query()` --calls--> `store_in_cache()`  [INFERRED]
  generation/main.py → cache/main.py
- `run_query()` --calls--> `merge_ranked_chunks()`  [INFERRED]
  generation/main.py → retrieval/chunk_utils.py
- `run_query()` --calls--> `get_embeddings()`  [INFERRED]
  generation/main.py → retrieval/embedder.py

## Import Cycles
- 1-file cycle: `api.py -> api.py`
- 1-file cycle: `ingestion/embedder.py -> ingestion/embedder.py`
- 1-file cycle: `retrieval/reranker.py -> retrieval/reranker.py`

## Communities (19 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.15
Nodes (16): custom_swagger_ui_html(), health_check(), lifespan(), query_endpoint(), QueryRequest, QueryResponse, Request model for query endpoint., Response model for query endpoint. (+8 more)

### Community 1 - "Community 1"
Cohesion: 0.18
Nodes (9): Document, extract_keys_from_schema(), _get_source(), process_all_files(), process_file(), convert_to_text(), split_text(), upsert_vectors() (+1 more)

### Community 2 - "Community 2"
Cohesion: 0.21
Nodes (11): initialize_search_pipeline(), get_dense_embedding(), get_embeddings(), get_sparse_embedding(), hybrid_score_norm(), load_dense_embedding_model(), load_sparse_embedding_model(), Load SPLADE tokenizer + raw ONNX session (+3 more)

### Community 3 - "Community 3"
Cohesion: 0.25
Nodes (13): check_pipeline_status(), fetch_langfuse_scores(), inject_global_styles(), load_employees(), main(), query_rag(), Fetch latest score per metric name from Langfuse, return {name: value}., Only inject styles that Streamlit won't strip — app-level chrome. (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.19
Nodes (8): check_cache(), store_in_cache(), build_context_block(), generate_response(), run_query(), merge_ranked_chunks(), retrieve(), rerank_local()

### Community 5 - "Community 5"
Cohesion: 0.47
Nodes (3): Pinecone, rerank_pinecone(), RerankResult

### Community 7 - "Community 7"
Cohesion: 0.50
Nodes (3): Logger, get_logger(), Logging configuration for EnterpriseRAG services.

## Knowledge Gaps
- **3 isolated node(s):** `HTMLResponse`, `Document`, `Logger`
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_query()` connect `Community 4` to `Community 0`, `Community 2`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `query_endpoint()` connect `Community 0` to `Community 4`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Why does `retrieve()` connect `Community 4` to `Community 8`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `run_query()` (e.g. with `query_endpoint()` and `check_cache()`) actually correct?**
  _`run_query()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `process_file()` (e.g. with `extract_keys_from_schema()` and `convert_to_text()`) actually correct?**
  _`process_file()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `initialize_search_pipeline()` (e.g. with `lifespan()` and `load_dense_embedding_model()`) actually correct?**
  _`initialize_search_pipeline()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `EnterpriseRAG package.`, `HTMLResponse`, `Request model for query endpoint.` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._