import os
from pathlib import Path
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from src.ingestion.config import EMBEDDING_MODEL, HF_TOKEN

MODEL_STORE = Path("model_store")

embedding_model = None

def load_embedding_model():
    global embedding_model
    
    if embedding_model is not None:
        return embedding_model

    model_name = EMBEDDING_MODEL.replace("/", "_")
    local_path = MODEL_STORE / model_name

    if local_path.exists():
        logger.info(f"Loading embedding model from {local_path}")
        embedding_model = SentenceTransformer(str(local_path))
    else:
        logger.info(f"Downloading embedding model {EMBEDDING_MODEL} ...")
        MODEL_STORE.mkdir(parents=True, exist_ok=True)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, token=HF_TOKEN)
        embedding_model.save(str(local_path))
        logger.info(f"Saved to {local_path}")

    return embedding_model

def get_dense_embedding(model, text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()


def get_sparse_embedding(pc: Pinecone, text: str) -> tuple:
    result = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=[text],
        parameters={"input_type": "passage", "truncate": "END"}
    )[0]
    return result["sparse_indices"], result["sparse_values"]