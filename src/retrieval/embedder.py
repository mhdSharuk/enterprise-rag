import os
import numpy as np
from pathlib import Path
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from src.utils.logger import logger
from src.retrieval.config import EMBEDDING_MODEL, HF_TOKEN

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
    logger.info(f'Creating dense embedding for text : {text[:30]}...')

    return model.encode(text, normalize_embeddings=True).tolist()


def get_sparse_embedding(pc: Pinecone, text: str):
    logger.info(f'Creating sparse embedding for text : {text[:30]}...')

    result = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=[text],
        parameters={"input_type": "query", "truncate": "END"}
    )[0]
    return result.sparse_indices, result.sparse_values


def hybrid_score_norm(dense: list, sparse_indices, sparse_values, alpha: float):
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")

    dense_arr = np.array(dense)

    if sparse_indices is not None:
        sparse_val_arr = np.array(sparse_values)
        h_dense = (dense_arr * alpha).tolist()
        h_sparse = {
            "indices": sparse_indices,
            "values": (sparse_val_arr * (1 - alpha)).tolist()
        }
        return h_dense, h_sparse
    else:
        return dense_arr.tolist(), {"indices": [], "values": []}