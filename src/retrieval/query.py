# from src.retrieval.reranker import rerank_pinecone, rerank_local
# from src.retrieval.config import HYBRID_ALPHA, RERANK_TOP_N, LOCAL_RERANK
# from src.retrieval.chunk_utils import merge_ranked_chunks, get_neighbour_chunk_ids
# from src.retrieval.vector_store import query_all_sources, fetch_vectors, get_pinecone_index
# from src.retrieval.embedder import get_dense_embedding, hybrid_score_norm, load_embedding_model, get_sparse_embedding

# def retrieve(query: str, embedding_model, pc, index,
#              use_sparse: bool = False,
#              rerank_top_n: int = RERANK_TOP_N) -> list[dict]:

#     dense = get_dense_embedding(embedding_model, query)

#     if use_sparse:
#         sparse_indices, sparse_values = get_sparse_embedding(pc, query)
#     else:
#         sparse_indices, sparse_values = None, None

#     hdense, hsparse = hybrid_score_norm(dense, sparse_indices, sparse_values, alpha=HYBRID_ALPHA)

#     documents = query_all_sources(index, hdense, hsparse if use_sparse else None)

#     # for doc in documents:
#     #     s = doc['score']
#     #     d = doc['id']
#     #     print(f'{s} => {d}')

#     if LOCAL_RERANK:
#         reranked = rerank_local(query, documents)
#     else:
#         reranked = rerank_pinecone(pc, query, documents, top_n=rerank_top_n)

#     print(reranked)

#     # merged = merge_ranked_chunks(reranked)

#     # neighbour_ids = list(get_neighbour_chunk_ids(reranked.data))
#     # refetched = fetch_vectors(index, neighbour_ids)

#     # extra_chunks = []
#     # for vec_id, vec_data in refetched.vectors.items():
#     #     if not any(m["id"] == vec_id for m in merged):
#     #         extra_chunks.append({
#     #             "id": vec_id,
#     #             "chunk_range": None,
#     #             "score": 0.0,
#     #             "text": vec_data.metadata.get("text", "")
#     #         })

#     # return merged + extra_chunks


# if __name__ == "__main__":

#     pc, index = get_pinecone_index()
#     embedding_model = load_embedding_model()

#     query = input("Enter your query: ").strip()
#     if not query:
#         print("No query provided.")
#         exit(1)

#     results = retrieve(query, embedding_model, pc, index)

#     # print(f"\nTop {len(results)} results:\n")
#     # for i, doc in enumerate(results, 1):
#     #     chunk_info = f"chunks {doc['chunk_range']}" if doc["chunk_range"] else "neighbour chunk"
#     #     print(f"[{i}] {doc['id']} ({chunk_info}) | score: {doc['score']:.4f}")
#     #     print(f"    {doc['text'][:200].strip()}...")
#     #     print()



from src.retrieval.reranker import rerank_pinecone, rerank_local
from src.retrieval.embedder import get_dense_embedding, hybrid_score_norm
from src.retrieval.vector_store import query_all_sources, fetch_vectors
from src.retrieval.chunk_utils import merge_ranked_chunks, get_neighbour_chunk_ids

from src.retrieval.config import HYBRID_ALPHA, RERANK_TOP_N, LOCAL_RERANK


def retrieve(query: str, embedding_model, pc, index,
             reranker=None,
             use_sparse: bool = False,
             rerank_top_n: int = RERANK_TOP_N) -> list[dict]:

    dense = get_dense_embedding(embedding_model, query)

    if use_sparse:
        from src.retrieval.embedder import get_sparse_embedding
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

    print(reranked)

    # merged = merge_ranked_chunks(reranked)

    # neighbour_ids = list(get_neighbour_chunk_ids(reranked.data))
    # refetched = fetch_vectors(index, neighbour_ids)

    # extra_chunks = []
    # for vec_id, vec_data in refetched.vectors.items():
    #     if not any(m["id"] == vec_id for m in merged):
    #         extra_chunks.append({
    #             "id": vec_id,
    #             "chunk_range": None,
    #             "score": 0.0,
    #             "text": vec_data.metadata.get("text", "")
    #         })

    # return merged + extra_chunks


if __name__ == "__main__":
    from src.retrieval.embedder import load_embedding_model
    from src.retrieval.vector_store import get_pinecone_index
    from src.retrieval.reranker import load_reranker

    pc, index = get_pinecone_index()
    embedding_model = load_embedding_model()
    reranker = load_reranker() if LOCAL_RERANK else None

    query = input("Enter your query: ").strip()
    if not query:
        print("No query provided.")
        exit(1)

    results = retrieve(query, embedding_model, pc, index, reranker=reranker)

    print(f"\nTop {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        chunk_info = f"chunks {doc['chunk_range']}" if doc["chunk_range"] else "neighbour chunk"
        print(f"[{i}] {doc['id']} ({chunk_info}) | score: {doc['score']:.4f}")
        print(f"    {doc['text'][:200].strip()}...")
        print()