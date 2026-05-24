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


def run_query(user_name, query, pc, pc_index, 
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
        user_name,
        pc_index = pc_index,
        query = query,
        query_dense_embedding = query_dense_embedding,
        query_sparse_embedding = query_sparse_embedding,
        reranker_tokenizer = reranker_model[0],
        reranker_model = reranker_model[1]
    )

    # cache_hit = False
    if cache_hit:
      print('Cache Hit')
      return (cache_hit, 
              cache_docs,
              cached_response,
              0,
              'Cache Hit')

    else:
      print(f'Cache Missed. Proceeding to LLM')

      retrieved_docs = retrieve(
          user_name=user_name,
          query=query,
          query_dense_embedding=query_dense_embedding,
          query_sparse_embedding=query_sparse_embedding,
          pc_index=pc_index,
          reranker=reranker_model
      )

      if not retrieved_docs.data:
        print('No permission')
        return (False, [], 
                'You dont have the permission to access the file',
                0, 'Access restricted')
      else:

        merged_docs = merge_ranked_chunks(retrieved_docs.data)
        sources = [doc['id'] for doc in merged_docs]

        answer, total_tokens, finish_reason = generate_response(query, merged_docs)
        user_access = merged_docs[0]['user_access']

        store_in_cache(pc_index, query, answer, 
              query_dense_embedding, 
              query_sparse_embedding,
              sources, user_access)

        return False, merged_docs, answer, total_tokens, finish_reason