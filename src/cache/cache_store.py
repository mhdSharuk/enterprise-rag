import hashlib
from pinecone import Pinecone
from src.cache.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

CACHE_NAMESPACE = "cache"

def get_cache_index():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        return pc, index

    except Exception as error:
        print(f'Error occured')
        print(error)
        return None, None

def store_in_vectorstore(index, query, response, dense_embedding, sparse_embedding=None):
    query_hash = hashlib.sha256(query.encode()).hexdigest()

    try:
        index.upsert(
            vectors=[{
                "id": query_hash,
                "values": dense_embedding,
                "metadata": {
                    "original_query": query,
                    "response": response,
                    "is_cache": True
                }
            }],
            namespace=CACHE_NAMESPACE
        )
        print(f"Stored in cache for query hash: {query_hash}")

    except Exception as e:
        print(f"Error storing in cache: {e}")

def search_in_vectorstore(index, vector: list, top_k: int = 5) -> list[dict]:
    try:
        response = index.query(
            vector=vector,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            namespace=CACHE_NAMESPACE,
            async_req=False
        )

        matches = response.to_dict()["matches"]

        results = []
        for m in matches:
            results.append({
                "id": m["id"],
                "score": m["score"],
                "response": m["metadata"]["response"],
                "cached_query": m["metadata"]["original_query"],
                "entities": m["metadata"].get("entities", [])
            })

        return results

    except Exception as e:
        print(f"Cache search error: {e}")
        return []