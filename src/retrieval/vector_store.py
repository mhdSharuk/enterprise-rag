from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

from src.utils.logger import logger
from src.retrieval.config import (PINECONE_API_KEY, PINECONE_INDEX_NAME, 
                                  SOURCES, TOP_K_PER_SOURCE, PINECONE_NAMESPACE)


def get_pinecone_index():
    try:
        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc, pc.Index(PINECONE_INDEX_NAME) 
    
    except Exception as error:
        logger.error(f'Error occured')
        logger.error(error)

        return None, None


def query_all_sources(index, dense_vector, 
                      sparse_vector = None, 
                      top_k = TOP_K_PER_SOURCE) -> list[dict]:
    futures = []
    for source in tqdm(SOURCES, desc="Querying sources"):        
        future = index.query(
            top_k=top_k,
            vector=dense_vector,
            sparse_vector=sparse_vector if sparse_vector and sparse_vector["indices"] else None,
            include_values=False,
            include_metadata=True,
            filter={"source": {"$eq": source}},
            async_req=True,
            namespace=PINECONE_NAMESPACE
        )
        futures.append(future)

    documents = []
    for future in tqdm(futures, desc="Collecting results"):
        response = future.get()
        matches = response.to_dict()["matches"]
        docs = [
            {
                "id": m["id"],
                "score": m["score"],
                "chunk_text": m["metadata"]["text"],
                'doc_id': m['metadata']['dataset_doc_uuid']
            }
            for m in matches
        ]
        documents.extend(docs)

    return sorted(documents, key=lambda x: x["score"], reverse=True)


def fetch_vectors_by_id(index, ids: list[str]) -> dict:

    documents = []
    futures = []
    
    for id_ in ids:
        future = index.query(
            id=id_,       
            top_k=1,                   
            include_values=False,  
            include_metadata=True,
            async_req=True,
            namespace=PINECONE_NAMESPACE
        )
        futures.append(future)

    for future in futures:
        response = future.get()
        matches = response.to_dict()["matches"]
        docs = [
            {
                "id": m["id"],
                "score": m["score"],
                "chunk_text": m["metadata"]["text"],
                'doc_id': m['metadata']['dataset_doc_uuid']
            }
            for m in matches
        ]
        documents.extend(docs)

    return documents
