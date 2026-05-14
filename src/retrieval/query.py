from src.retrieval.reranker import rerank_pinecone, rerank_local, load_reranker
from src.retrieval.chunk_utils import merge_ranked_chunks, get_neighbour_chunk_ids
from src.retrieval.vector_store import query_all_sources, fetch_vectors_by_id, get_pinecone_index
from src.retrieval.embedder import get_dense_embedding, hybrid_score_norm, load_embedding_model, get_sparse_embedding

from src.retrieval.config import HYBRID_ALPHA, RERANK_TOP_N, LOCAL_RERANK

def retrieve(query, 
            embedding_tokenizer,
            embedding_model, 
            pc, index,
            reranker = None,
            use_sparse = False,
            rerank_top_n = RERANK_TOP_N) -> list[dict]:

    # dense = get_dense_embedding(embedding_model, query)
    dense = get_dense_embedding(embedding_tokenizer, embedding_model, query)

    if use_sparse:
        sparse_indices, sparse_values = get_sparse_embedding(pc, query)
    else:
        sparse_indices, sparse_values = None, None

    hdense, hsparse = hybrid_score_norm(dense, sparse_indices, sparse_values, alpha=HYBRID_ALPHA)

    documents = query_all_sources(index, hdense, hsparse if use_sparse else None)

    if LOCAL_RERANK:
        tokenizer, model = reranker
        reranked = rerank_local(tokenizer, model, query, documents)
    else:
        reranked = rerank_pinecone(pc, query, documents, top_n=rerank_top_n)

    neighbour_ids = list(get_neighbour_chunk_ids(reranked.data))
    refetched = fetch_vectors_by_id(index, neighbour_ids)
    merged = merge_ranked_chunks(refetched)

    return merged


if __name__ == "__main__":

    pc, index = get_pinecone_index()
    dense_embedding_tokenizer, dense_embedding_model = load_embedding_model()
    reranker = load_reranker() if LOCAL_RERANK else None

    query = input("Enter your query: ").strip()
    if not query:
        print("No query provided.")
        exit(1)

    results = retrieve(query, 
                    dense_embedding_tokenizer, 
                    dense_embedding_model,
                    pc, index, 
                    reranker=reranker)

    print(results)