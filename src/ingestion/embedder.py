import numpy as np
import onnxruntime as ort

from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForMaskedLM

from src.utils.logger import logger
from src.utils.check_device import get_onnx_provider
from src.ingestion.config import (DENSE_EMBEDDING_MODEL, DENSE_EMBEDDING_ONNX_FILE,
                                  SPARSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_ONNX_FILE,
                                  HF_TOKEN, HYBRID_ALPHA)

MODEL_STORE = Path("model_store")
DENSE_EMBEDDER_PATH = MODEL_STORE / DENSE_EMBEDDING_MODEL.replace("/", "_")
SPARSE_EMBEDDER_PATH = MODEL_STORE / SPARSE_EMBEDDING_MODEL.replace("/", "_")

embedding_tokenizer = None
embedding_model = None
splade_tokenizer = None
splade_session = None
splade_input_names = None

def load_dense_embedding_model():
    global embedding_tokenizer, embedding_model

    if embedding_model is not None:
        return embedding_tokenizer, embedding_model

    provider = get_onnx_provider()

    if DENSE_EMBEDDER_PATH.exists():
        logger.info(f"Loading ONNX embedding model from {DENSE_EMBEDDER_PATH}")

        embedding_tokenizer = AutoTokenizer.from_pretrained(
            str(DENSE_EMBEDDER_PATH),
            fix_mistral_regex=True
        )

        embedding_model = ORTModelForFeatureExtraction.from_pretrained(
            str(DENSE_EMBEDDER_PATH),
            file_name=DENSE_EMBEDDING_ONNX_FILE,
            token=HF_TOKEN,
            provider=provider
        )
        return embedding_tokenizer, embedding_model

    logger.info(f"Exporting embedding model {DENSE_EMBEDDING_MODEL} to ONNX ...")
    DENSE_EMBEDDER_PATH.mkdir(parents=True, exist_ok=True)

    embedding_tokenizer = AutoTokenizer.from_pretrained(
        DENSE_EMBEDDING_MODEL,
        token=HF_TOKEN,
        fix_mistral_regex=True
    )

    embedding_model = ORTModelForFeatureExtraction.from_pretrained(
        DENSE_EMBEDDING_MODEL,
        subfolder="onnx",
        file_name=DENSE_EMBEDDING_ONNX_FILE,
        token=HF_TOKEN,
        provider=provider
    )

    embedding_model.save_pretrained(str(DENSE_EMBEDDER_PATH))
    embedding_tokenizer.save_pretrained(str(DENSE_EMBEDDER_PATH))

    logger.info(f"Saved ONNX embedding model to {DENSE_EMBEDDER_PATH}")
    return embedding_tokenizer, embedding_model


def load_sparse_embedding_model():
    """Load SPLADE tokenizer + raw ONNX session"""
    global splade_tokenizer, splade_session, splade_input_names

    if splade_session is not None:
        return splade_tokenizer, splade_session, splade_input_names

    provider = get_onnx_provider()

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4

    if SPARSE_EMBEDDER_PATH.exists():
        logger.info(f"Loading ONNX SPLADE model from {SPARSE_EMBEDDER_PATH}")

        splade_tokenizer = AutoTokenizer.from_pretrained(str(SPARSE_EMBEDDER_PATH))

        onnx_path = str(SPARSE_EMBEDDER_PATH / SPARSE_EMBEDDING_ONNX_FILE)

        splade_session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=[provider]
        )

        splade_input_names = [inp.name for inp in splade_session.get_inputs()]

        return splade_tokenizer, splade_session, splade_input_names

    logger.info(f"Exporting SPLADE model {SPARSE_EMBEDDING_MODEL} to ONNX ...")
    SPARSE_EMBEDDER_PATH.mkdir(parents=True, exist_ok=True)

    splade_tokenizer = AutoTokenizer.from_pretrained(SPARSE_EMBEDDING_MODEL, token=HF_TOKEN)

    splade_model = ORTModelForMaskedLM.from_pretrained(
        SPARSE_EMBEDDING_MODEL,
        subfolder="onnx",
        file_name=SPARSE_EMBEDDING_ONNX_FILE,
        token=HF_TOKEN,
        provider=provider,
    )

    splade_model.save_pretrained(str(SPARSE_EMBEDDER_PATH))
    splade_tokenizer.save_pretrained(str(SPARSE_EMBEDDER_PATH))

    splade_session = ort.InferenceSession(
        str(SPARSE_EMBEDDER_PATH / SPARSE_EMBEDDING_ONNX_FILE),
        sess_options=session_options,
        providers=[provider]
    )

    splade_input_names = [inp.name for inp in splade_session.get_inputs()]

    logger.info(f"Saved and loaded SPLADE ONNX model")
    return splade_tokenizer, splade_session, splade_input_names


def get_dense_embedding(tokenizer, model, text):

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

    return embeddings


def get_sparse_embedding(tokenizer, session, input_names, text):
    logger.info(f'Creating sparse embedding for text : {text[:30]}...')

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )

    onnx_inputs = {}
    for onnx_name in input_names:
        if onnx_name == "input_ids":
            onnx_inputs[onnx_name] = inputs["input_ids"]
        elif onnx_name in ("attention_mask", "input_mask"):
            onnx_inputs[onnx_name] = inputs["attention_mask"]
        elif onnx_name in ("token_type_ids", "segment_ids"):
            if "token_type_ids" in inputs:
                onnx_inputs[onnx_name] = inputs["token_type_ids"]

    outputs = session.run(None, onnx_inputs)
    logits = outputs[0]

    activated = np.log1p(np.maximum(logits, 0.0))
    sparse_vector = np.max(activated, axis=1)[0]

    nonzero_indices = np.where(sparse_vector > 0.0)[0]
    nonzero_values = sparse_vector[nonzero_indices]

    sparse_indices = [int(idx) for idx in nonzero_indices]
    sparse_values = [float(val) for val in nonzero_values]

    return sparse_indices, sparse_values


def hybrid_score_norm(dense, sparse_indices, sparse_values, alpha):
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")

    dense_arr = np.array(dense)
    sparse_val_arr = np.array(sparse_values)
    h_dense = (dense_arr * alpha).tolist()
    h_sparse = {
        "indices": sparse_indices,
        "values": (sparse_val_arr * (1 - alpha)).tolist()
    }
    return h_dense, h_sparse


def get_embeddings(dense_embedding_tokenizer,
                   dense_embedding_model,
                   sparse_embedding_tokenizer,
                   sparse_embedding_model,
                   sparse_input_names,
                   text):

    dense_embedding = get_dense_embedding(dense_embedding_tokenizer, dense_embedding_model, text)
    sparse_indices, sparse_values = get_sparse_embedding(sparse_embedding_tokenizer,
                                                         sparse_embedding_model,
                                                         sparse_input_names,
                                                         text)

    hdense, hsparse = hybrid_score_norm(dense_embedding, sparse_indices, sparse_values, alpha=HYBRID_ALPHA)

    return hdense, hsparse
