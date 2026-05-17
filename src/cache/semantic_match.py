from src.retrieval.embedder import get_dense_embedding
from src.cache.cache_store import search_in_vectorstore

def check_semantic_match(index, query: str, embedding_tokenizer, embedding_model, threshold: float = 0.9) -> dict | None:
    embedding = get_dense_embedding(embedding_tokenizer, embedding_model, query)

    results = search_in_vectorstore(index, embedding, top_k=1)

    if results and len(results) > 0:
        best_match = results[0]
        if best_match["score"] >= threshold:
            return best_match

    return None