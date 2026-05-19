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

    print(f'Number of docs : {len(merged_docs)}')
    for docs in merged_docs:
        print(f'{docs["score"]} => {docs["id"]}')

    answer = f'Answer generated for query : {query}'#generate(query, merged_docs, tokenizer, model)

    # print(merged_docs[0])
    # print(merged_docs[0]['text'])

    # store_in_cache(index, query, answer, dense_embedding, sparse_embedding)

    # return answer


if __name__ == "__main__":
    # query = "What are the specific gate thresholds used to automatically decide whether a compressed model variant is allowed, canaried, or blocked, including the limits for chat similarity drop, code pass rate change, retrieval embedding quality, and acceptable latency and cost changes?"
    # query = "In the draft spec about extending a routing policy engine for automated regional failover, what is the proposed priority order for evaluating different failure signals when deciding whether to shift traffic or fail over?"
    query = "When is the 60 to 90 minute technical deep dive scheduled with the healthcare client about running model serving inside their own isolated network, and what is the time window in Pacific time?"
    # query = "In the notes about keeping long, stop-and-go chat sessions cheap without replaying the whole history, what storage setup and time-to-live were proposed for keeping the compact per-session state for recent sessions versus longer retention?"

    start_time = time.perf_counter()
    answer = run_query(query)
    end_time = time.perf_counter()

    print(f'Time taken: {end_time - start_time:.2f} seconds')

    # print("\n=== Answer ===")
    # print(answer)