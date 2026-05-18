import sys
import time

from src.retrieval.config import LOCAL_RERANK, HYBRID_ALPHA
from src.retrieval.reranker import load_local_reranker
from src.retrieval.embedder import load_embedding_model, get_embeddings
from src.retrieval.vector_store import get_pinecone_index
from src.retrieval.query import retrieve
from src.generation.model_loader import load_model
from src.generation.generator import generate
from src.cache.main import check_cache, store_in_cache


def run_query(query) -> str:
    pc, index = get_pinecone_index()
    dense_embedding_tokenizer, dense_embedding_model = load_embedding_model()
    reranker = load_local_reranker() if LOCAL_RERANK else None

    # tokenizer, model = load_model()

    query_dense_embedding, query_sparse_embedding = get_embeddings(
        dense_embedding_tokenizer, 
        dense_embedding_model, 
        pc, query
    )

    # hit, cached_response = check_cache(
    #     query = query,
    #     index = index,
    #     query_embedding = query_dense_embedding
    # )

    # if hit:
    #     return cached_response

    merged_docs, dense_embedding, sparse_embedding = retrieve(
        query=query,
        query_dense_embedding=query_dense_embedding,
        query_sparse_embedding=query_sparse_embedding,
        pc=pc,
        index=index,
        reranker=reranker
    )

    answer = f'Answer generated for query : {query}'#generate(query, merged_docs, tokenizer, model)


    store_in_cache(index, query, answer, dense_embedding, sparse_embedding)

    return answer


if __name__ == "__main__":
    query = "In the draft specification regarding the extended routing policy engine for automated regional failover, how are the various failure signals prioritized for traffic shifting and failover decisions?"
    
    start_time = time.perf_counter()
    answer = run_query(query)
    end_time = time.perf_counter()

    print(f'Time taken: {end_time - start_time:.2f} seconds')

    print("\n=== Answer ===")
    print(answer)