import os
import numpy as np
from pathlib import Path
from pinecone import Pinecone
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer, logging as transformers_logging

# from sentence_transformers import SentenceTransformer

from src.utils.logger import logger
from src.utils.check_device import get_onnx_provider
from src.retrieval.config import (EMBEDDING_MODEL, HF_TOKEN, EMBEDDING_ONNX_FILE)

MODEL_STORE = Path("model_store")
EMBEDDER_PATH = MODEL_STORE / EMBEDDING_MODEL.replace("/", "_")

embedding_tokenizer = None
embedding_model = None

def load_embedding_model():
    global embedding_tokenizer, embedding_model

    if embedding_model is not None:
        return embedding_tokenizer, embedding_model

    provider = get_onnx_provider()

    if EMBEDDER_PATH.exists():
        logger.info(f"Loading ONNX embedding model from {EMBEDDER_PATH}")
        
        embedding_tokenizer = AutoTokenizer.from_pretrained(
          str(EMBEDDER_PATH),
          fix_mistral_regex=True
        )
        
        local_file = "model_quantized.onnx" if (EMBEDDER_PATH / "model_quantized.onnx").exists() else "model.onnx"
        embedding_model = ORTModelForFeatureExtraction.from_pretrained(
            str(EMBEDDER_PATH),
            file_name = local_file,
            token = HF_TOKEN,
            provider=provider
        )
        return embedding_tokenizer, embedding_model 

    logger.info(f"Exporting embedding model {EMBEDDING_MODEL} to ONNX ...")
    EMBEDDER_PATH.mkdir(parents=True, exist_ok=True)

    embedding_tokenizer = AutoTokenizer.from_pretrained(
      EMBEDDING_MODEL, 
      token=HF_TOKEN,
      fix_mistral_regex=True
    )
    
    embedding_model = ORTModelForFeatureExtraction.from_pretrained(
        EMBEDDING_MODEL,
        subfolder="onnx",
        file_name = EMBEDDING_ONNX_FILE,
        token=HF_TOKEN,
        provider=provider
    )

    embedding_model.save_pretrained(str(EMBEDDER_PATH))
    embedding_tokenizer.save_pretrained(str(EMBEDDER_PATH))

    logger.info(f"Saved ONNX embedding model to {EMBEDDER_PATH}")
    return embedding_tokenizer, embedding_model

def get_dense_embedding(tokenizer, model, text: str) -> list[float]:

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        return np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(input_mask_expanded.sum(1), 1e-9)


    logger.info(f'Creating dense embedding for text : {text[:30]}...')

    inputs = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="np"
    )

    outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norm

    return embeddings[0].tolist()


# def load_embedding_model():
#     global embedding_model
    
#     if embedding_model is not None:
#         return embedding_model

#     model_name = EMBEDDING_MODEL.replace("/", "_")
#     local_path = MODEL_STORE / model_name

#     if local_path.exists():
#         logger.info(f"Loading embedding model from {local_path}")
#         embedding_model = SentenceTransformer(str(local_path))
#     else:
#         logger.info(f"Downloading embedding model {EMBEDDING_MODEL} ...")
#         MODEL_STORE.mkdir(parents=True, exist_ok=True)
#         embedding_model = SentenceTransformer(EMBEDDING_MODEL, token=HF_TOKEN)
#         embedding_model.save(str(local_path))
#         logger.info(f"Saved to {local_path}")

#     return embedding_model

# def get_dense_embedding(model, text: str) -> list[float]:
#     logger.info(f'Creating dense embedding for text : {text[:30]}...')

#     return model.encode(text, normalize_embeddings=True).tolist()


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