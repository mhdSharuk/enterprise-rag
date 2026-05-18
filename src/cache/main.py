from src.utils.logger import logger
from src.retrieval.embedder import get_dense_embedding
from src.cache.cache_store import store_in_vectorstore
from src.cache.query_validator import compute_bm25_scores, compute_hybrid_score, check_semantic_match
from src.cache.config import CACHE_FINAL_THRESHOLD


def check_cache(query, index, query_embedding):

    semantic_results = check_semantic_match(index, query, query_embedding)

    if not semantic_results:
        logger.info("Cache miss: no semantic matches above threshold")
        return False, None

    cached_queries = [r["cached_query"] for r in semantic_results]
    bm25_scores = compute_bm25_scores(query, cached_queries)

    best_final_score = 0.0
    best_response = None

    for i, result in enumerate(semantic_results):
        semantic_score = result["score"]
        bm25_score = bm25_scores[i]
        final_score = compute_hybrid_score(semantic_score, bm25_score)

        if final_score > best_final_score:
            best_final_score = final_score
            best_response = result["response"]

    if best_final_score >= CACHE_FINAL_THRESHOLD:
        logger.info(f"Cache hit: hybrid match (score: {best_final_score:.3f})")
        return True, best_response
    else:
        logger.info(f"Cache miss: hybrid score {best_final_score:.3f} below threshold")
        return False, None

def store_in_cache(index, query, response, query_dense_embedding, query_sparse_embedding=None):

    store_in_vectorstore(index, query, response, query_dense_embedding)
    logger.info(f"Stored in cache: {query[:50]}...")