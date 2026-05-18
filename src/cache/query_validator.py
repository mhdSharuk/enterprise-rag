import math
from rank_bm25 import BM25Okapi

from src.cache.config import (
    BM25_SIGMOID_SCALE,
    CACHE_SEMANTIC_THRESHOLD,
    CACHE_SEMANTIC_TOP_K,
    HYBRID_SCORE_ALPHA
)
from src.cache.cache_store import search_in_vectorstore
from src.retrieval.embedder import get_dense_embedding


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def compute_bm25_scores(query: str, documents: list[str]) -> list[float]:
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    return scores.tolist()


def sigmoid(score: float, scale: float = BM25_SIGMOID_SCALE) -> float:
    return 1 / (1 + math.exp(-scale * score))


def compute_hybrid_score(semantic_score: float, bm25_score: float) -> float:
    bm25_sigmoid = sigmoid(bm25_score)
    hybrid_score = HYBRID_SCORE_ALPHA * semantic_score + (1 - HYBRID_SCORE_ALPHA) * bm25_sigmoid
    return hybrid_score


def check_semantic_match(index, query, query_embedding) -> list[dict]:
    results = search_in_vectorstore(index, query_embedding, top_k=CACHE_SEMANTIC_TOP_K)
    filtered_results = [
        r for r in results if r["score"] >= CACHE_SEMANTIC_THRESHOLD
    ]
    return filtered_results