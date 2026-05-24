import hashlib
import traceback
from src.utils.logger import logger
from src.cache.config import (
    CACHE_SCORE_THRESHOLD,
    CACHE_SEMANTIC_TOP_K,
    PINECONE_CACHE_NAMESPACE
)
from pprint import pprint
from src.retrieval.reranker import rerank_local

def check_cache(user_name, pc_index, query, 
                query_dense_embedding, 
                query_sparse_embedding,
                reranker_tokenizer, 
                reranker_model):

    try:
        response = pc_index.query(
            vector=query_dense_embedding,
            sparse_vector=query_sparse_embedding,
            top_k=CACHE_SEMANTIC_TOP_K,
            include_values=False,
            include_metadata=True,
            filter={"user_access": {"$in": [user_name]}},
            namespace=PINECONE_CACHE_NAMESPACE,
            async_req=False
        )

        matches = response.to_dict()["matches"]

        if matches:
          cache_results = []
          for m in matches:
              cache_results.append({
                  "id": m["id"],
                  "score": m["score"],
                  "cached_query": m["metadata"]["query"],
                  "answer": m["metadata"]["answer"],
                  "source_deps": m['metadata']['source_deps'],
                  "user_access": m['metadata']['user_access']
              })
    
        
          cache_reranked = rerank_local(reranker_tokenizer, 
                                      reranker_model, 
                                      query, 
                                      cache_results, 
                                      text_field='cached_query')

  
          cache_reranked_filtered = [res for res in cache_reranked.data if res['score'] >= CACHE_SCORE_THRESHOLD]

          if cache_reranked_filtered:
              return (True, cache_reranked_filtered, 
                      cache_reranked_filtered[0]["response"])
          else:
              return False, [], None

        else:
          return False, [], None

    except Exception as err:
        logger.error(f"Error in check_cache")
        traceback.print_exc()
        return False, [], None

def store_in_cache(pc_index, query, response, 
                   query_dense_embedding, 
                   query_sparse_embedding,
                   sources, user_access):

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
                    "source_deps": sources,
                    "user_access": user_access
                }
            }],
            namespace=PINECONE_CACHE_NAMESPACE
        )
        print(f"Stored in cache for query hash: {query_hash}")

    except Exception as e:
        print(f"Error storing in cache: {e}")
