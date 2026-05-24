# import time
from src.retrieval.vector_store import query_all_sources
from src.retrieval.reranker import rerank_local
# from src.retrieval.embedder import (get_dense_embedding, 
#                                     get_sparse_embedding,
#                                     hybrid_score_norm, 
#                                     )

from src.retrieval.config import HYBRID_ALPHA, RERANK_TOP_N, LOCAL_RERANK

def retrieve(user_name, query,
            query_dense_embedding,
            query_sparse_embedding,
            pc_index,
            reranker) -> list[dict]:

    
    documents = query_all_sources(pc_index, query_dense_embedding, query_sparse_embedding, user_name)

    rerank_tokenizer, rerank_model = reranker
    reranked = rerank_local(rerank_tokenizer, rerank_model, query, documents)

    return reranked

