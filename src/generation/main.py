import time

from src.retrieval.query import retrieve
from src.retrieval.config import LOCAL_RERANK
from src.cache.main import check_cache, store_in_cache
from src.generation.generator import generate_response
from src.retrieval.reranker import load_local_reranker
from src.retrieval.chunk_utils import merge_ranked_chunks
from src.retrieval.vector_store import get_pinecone_index
from src.retrieval.embedder import (get_embeddings, load_dense_embedding_model,
                                    load_sparse_embedding_model)


def initialize_search_pipeline():
    pc, pc_index = get_pinecone_index()
    dense_tokenizer, dense_model = load_dense_embedding_model()
    sparse_tokenizer, sparse_model, sparse_input_names = load_sparse_embedding_model()
    reranker = load_local_reranker() if LOCAL_RERANK else None
    
    return (pc, pc_index, dense_tokenizer, dense_model, 
            sparse_tokenizer, sparse_model, 
            sparse_input_names, reranker)


def run_query(query, pc, pc_index, 
            dense_tokenizer, dense_model, 
            sparse_tokenizer, sparse_model,
            sparse_input_names, reranker_model) -> tuple:

    query_dense_embedding, query_sparse_embedding = get_embeddings(
        dense_tokenizer, 
        dense_model, 
        sparse_tokenizer,
        sparse_model,
        sparse_input_names,
        query
    )

    cache_hit, cache_docs, cached_response = check_cache(
        pc_index = pc_index,
        query = query,
        query_dense_embedding = query_dense_embedding,
        query_sparse_embedding = query_sparse_embedding,
        reranker_tokenizer = reranker_model[0],
        reranker_model = reranker_model[1]
    )

    if cache_hit:
      return (cache_hit, 
              cache_docs,
              cached_response,
              0,
              'Cache Hit')

    else:
      print(f'Cache Missed. Proceeding to LLM')

      retrieved_docs = retrieve(
          query=query,
          query_dense_embedding=query_dense_embedding,
          query_sparse_embedding=query_sparse_embedding,
          pc_index=pc_index,
          reranker=reranker_model
      )

      merged_docs = merge_ranked_chunks(retrieved_docs.data)
      sources = [doc['id'] for doc in merged_docs]

      answer, total_tokens, finish_reason = generate_response(query, merged_docs)

      store_in_cache(pc_index, query, answer, 
            query_dense_embedding, 
            query_sparse_embedding,
            sources)

      return False, merged_docs, answer, total_tokens, finish_reason

if __name__ == "__main__":
    (pc, pc_index, dense_tokenizer, dense_model, 
     sparse_tokenizer, sparse_model, sparse_input_names, 
     reranker) = initialize_search_pipeline()

    # query = "What are the specific gate thresholds used to automatically decide whether a compressed model variant is allowed, canaried, or blocked, including the limits for chat similarity drop, code pass rate change, retrieval embedding quality, and acceptable latency and cost changes?"
    # query = "In the draft spec about extending a routing policy engine for automated regional failover, what is the proposed priority order for evaluating different failure signals when deciding whether to shift traffic or fail over?"
    # query = "When is the 60 to 90 minute technical deep dive scheduled with the healthcare client about running model serving inside their own isolated network, and what is the time window in Pacific time?"
    # query = "In the notes about keeping long, stop-and-go chat sessions cheap without replaying the whole history, what storage setup and time-to-live were proposed for keeping the compact per-session state for recent sessions versus longer retention?"

    # query = "What are the default size limits for file uploads and total request size for the new multipart upload support on the OpenAI-compatible API endpoints?"
    query = "What is the name of the new metric added so SRE can track when server-side streaming sessions get finalized due to hitting the time limit?"

    start_time = time.perf_counter()
    (is_cache_hit, retrieved_docs, 
    answer, total_tokens, 
    finish_reason) = run_query(query, pc, pc_index, 
                                dense_tokenizer, dense_model, 
                                sparse_tokenizer, sparse_model, 
                                sparse_input_names, reranker)

    # retrieved_docs = run_query(query[1], pc, pc_index, 
    #           dense_tokenizer, dense_model, 
    #           sparse_tokenizer, sparse_model, 
    #           sparse_input_names, reranker)

    end_time = time.perf_counter()

    print(f'Time taken: {end_time - start_time:.2f} seconds')
    print(f'Is Cache hit : {is_cache_hit}')
    print('Total tokens used:', total_tokens)
    print('Finish reason:', finish_reason)

    print("\n=== Answer ===")
    print(answer)