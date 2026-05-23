import hashlib
from src.utils.logger import logger
from src.cache.config import (
    CACHE_SCORE_THRESHOLD,
    CACHE_SEMANTIC_TOP_K,
    PINECONE_CACHE_NAMESPACE
)


def check_cache(pc_index, query_dense_embedding, query_sparse_embedding):

    try:
        response = pc_index.query(
            vector=query_dense_embedding,
            sparse_vector=query_sparse_embedding,
            top_k=CACHE_SEMANTIC_TOP_K,
            include_values=False,
            include_metadata=True,
            namespace=PINECONE_CACHE_NAMESPACE,
            async_req=False
        )

        matches = response.to_dict()["matches"]

        cache_results = []
        for m in matches:
            cache_results.append({
                "id": m["id"],
                "score": m["score"],
                "cached_query": m["metadata"]["query"],
                "response": m["metadata"]["answer"],
            })
    
        filtered_results = [
            result for result in cache_results if result["score"] >= CACHE_SCORE_THRESHOLD
        ]

        if not filtered_results:
            logger.info("Cache miss: no semantic matches above threshold")
            return False, None

        return True, filtered_results[0]["response"]

    except Exception as err:
        print(f"Error in check_cache : {err}")
        return False, None

def store_in_cache(pc_index, query, response, 
                   query_dense_embedding, 
                   query_sparse_embedding,
                   sources):

    query_hash = hashlib.sha256(query.encode()).hexdigest()

    try:
        pc_index.upsert(
            vectors=[{
                "id": query_hash,
                "values": query_dense_embedding,
                "sparse_values": query_sparse_embedding,
                "metadata": {
                    'answer': response,
                    "query": query,
                    "source_deps": True
                }
            }],
            namespace=PINECONE_CACHE_NAMESPACE
        )
        print(f"Stored in cache for query hash: {query_hash}")

    except Exception as e:
        print(f"Error storing in cache: {e}")