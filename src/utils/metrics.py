from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total query requests",
    ["status"]  # "success" | "error"
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "End-to-end query latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
)

CACHE_HITS = Counter("rag_cache_hits_total", "Semantic cache hits")
CACHE_MISSES = Counter("rag_cache_misses_total", "Semantic cache misses")

EMBEDDING_LATENCY = Histogram(
    "rag_embedding_latency_seconds",
    "Time to compute dense + sparse embeddings",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

RERANKER_LATENCY = Histogram(
    "rag_reranker_latency_seconds",
    "Time for cross-encoder reranking",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PINECONE_LATENCY = Histogram(
    "rag_pinecone_latency_seconds",
    "Time for Pinecone vector query",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

ERROR_COUNT = Counter(
    "rag_errors_total",
    "Errors by component",
    ["component"]  # "embedding" | "pinecone" | "reranker" | "generation" | "cache"
)
