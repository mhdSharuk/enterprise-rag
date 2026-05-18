import time
from src.retrieval.chunk_utils import merge_ranked_chunks
from src.retrieval.vector_store import query_all_sources, get_pinecone_index
from src.retrieval.reranker import rerank_pinecone, rerank_local, load_local_reranker
from src.retrieval.embedder import get_dense_embedding, hybrid_score_norm, load_embedding_model, get_sparse_embedding

from src.retrieval.config import HYBRID_ALPHA, RERANK_TOP_N, LOCAL_RERANK

def retrieve(query,
            query_dense_embedding,
            query_sparse_embedding,
            pc, index,
            reranker = None,
            use_sparse = False,
            rerank_top_n = RERANK_TOP_N) -> list[dict]:

    
    documents = query_all_sources(index, query_dense_embedding, query_sparse_embedding)

    if LOCAL_RERANK:
        tokenizer, model = reranker
        reranked = rerank_local(tokenizer, model, query, documents)
    else:
        reranked = rerank_pinecone(pc, query, documents, top_n=rerank_top_n)

    merged = merge_ranked_chunks(reranked.data)

    return merged, query_dense_embedding, query_sparse_embedding


# if __name__ == "__main__":

#     pc, index = get_pinecone_index()
#     dense_embedding_tokenizer, dense_embedding_model = load_embedding_model()
#     reranker = load_local_reranker() if LOCAL_RERANK else None

#     query = input("Enter your query: ").strip()
#     if not query:
#         print("No query provided.")
#         exit(1)

#     query = query.strip()

#     start_time = time.perf_counter()
#     results, dense_embedding, sparse_embedding = retrieve(query, 
#                     dense_embedding_tokenizer, 
#                     dense_embedding_model,
#                     pc, index, 
#                     reranker=reranker)
#     end_time = time.perf_counter()


#     print(f'Time taken: {end_time - start_time:.2f} seconds')
#     print('Results : ')
#     for doc in results:
#         print(f" - {doc['id']}")

