from pinecone import Pinecone, ServerlessSpec
from src.ingestion.config import (PINECONE_API_KEY, PINECONE_INDEX_NAME,
                                  PINECONE_DIMENSION, PINECONE_METRIC, 
                                  PINECONE_REGION, PINECONE_CLOUD)

def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )

    return pc, pc.Index(PINECONE_INDEX_NAME)


def upsert_vectors(index, vectors: list[dict]):
    if vectors:
        index.upsert(vectors=vectors)