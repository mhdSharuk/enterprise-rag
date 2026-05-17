import sys

from src.retrieval.config import LOCAL_RERANK
from src.retrieval.reranker import load_reranker
from src.retrieval.embedder import load_embedding_model
from src.retrieval.vector_store import get_pinecone_index
from src.retrieval.query import retrieve
from src.generation.model_loader import load_model
from src.generation.generator import generate
from src.cache.main import check_cache, store_in_cache


def run_query(query: str) -> str:
    pc, index = get_pinecone_index()
    dense_embedding_tokenizer, dense_embedding_model = load_embedding_model()
    reranker = load_reranker() if LOCAL_RERANK else None

    # tokenizer, model = load_model()

    hit, cached_response = check_cache(
        query=query,
        index=index,
        embedding_tokenizer=dense_embedding_tokenizer,
        embedding_model=dense_embedding_model
    )

    if hit:
        return cached_response

    # merged_docs = retrieve(
    #     query=query,
    #     embedding_tokenizer=dense_embedding_tokenizer,
    #     embedding_model=dense_embedding_model,
    #     pc=pc,
    #     index=index,
    #     reranker=reranker
    # )

    answer = f'Answer generated for query : {query}'#generate(query, merged_docs, tokenizer, model)

    store_in_cache(index, query, answer, dense_embedding_tokenizer, dense_embedding_model)

    return answer


if __name__ == "__main__":
    query = (
        'In the draft spec about extending a routing policy engine for automated '
        'regional failover, what is the proposed priority order for evaluating '
        'different failure signals when deciding whether to shift traffic or fail over?'
    )

    answer = run_query(query)
    print("\n=== Answer ===")
    print(answer)