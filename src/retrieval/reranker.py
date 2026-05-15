import numpy as np
from pathlib import Path
from pinecone import Pinecone
from optimum.utils import logging as optimum_logging
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, logging as transformers_logging

from src.utils.logger import logger
from src.utils.check_device import get_onnx_provider
from src.retrieval.config import (RERANKING_MODEL, RERANKING_ONNX_FILE, 
                                  HF_TOKEN, RERANK_TOP_N, RERANK_THRESHOLD)

MODEL_STORE = Path("model_store")
RERANKER_PATH = MODEL_STORE / RERANKING_MODEL.replace("/", "_")

class RerankResult:
    def __init__(self, model_name: str, documents: list, scores: list):
        self.model = model_name        
        self.data = sorted(
            [
                {
                    "score"      : float(score), 
                    "id"         : docs["id"], 
                    "chunk_text" : docs["chunk_text"],
                    'doc_id'     : docs['doc_id']
                }
                for score, docs in zip(scores, documents)
                if score >= RERANK_THRESHOLD
            ],
            key=lambda x: x["score"],
            reverse=True
        )

    def __repr__(self):
        return f"RerankResult(model='{self.model}', data={self.data})"



def load_reranker():
    provider = get_onnx_provider()

    if RERANKER_PATH.exists():
        logger.info(f"Loading reranker from {RERANKER_PATH}")

        tokenizer = AutoTokenizer.from_pretrained(
          str(RERANKER_PATH),
          fix_mistral_regex=True
        )

        local_file = "model_quantized.onnx" if (RERANKER_PATH / "model_quantized.onnx").exists() else "model.onnx"
        model = ORTModelForSequenceClassification.from_pretrained(
            str(RERANKER_PATH),
            file_name=local_file,
            token=HF_TOKEN,
            provider=provider
        )
        return tokenizer, model

    logger.info(f"Exporting reranker {RERANKING_MODEL} to ONNX ...")
    MODEL_STORE.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
      RERANKING_MODEL, 
      token=HF_TOKEN,
      fix_mistral_regex=True
    )
    model = ORTModelForSequenceClassification.from_pretrained(
        RERANKING_MODEL,
        subfolder="onnx",
        file_name = RERANKING_ONNX_FILE,
        token=HF_TOKEN,
        provider=provider
    )

    model.save_pretrained(str(RERANKER_PATH))
    tokenizer.save_pretrained(str(RERANKER_PATH))

    logger.info(f"Saved ONNX reranker to {RERANKER_PATH}")

    return tokenizer, model


def rerank_local(tokenizer, model, query: str, documents: list[dict], batch_size: int = 4) -> RerankResult:
    all_scores = []
    doc_texts = [doc["chunk_text"] for doc in documents]

    for i in range(0, len(doc_texts), batch_size):
        batch_docs = doc_texts[i : i + batch_size]
        
        inputs = tokenizer(
            [query] * len(batch_docs),
            batch_docs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np" 
        )

        outputs = model(**inputs)
        
        logits = outputs.logits.flatten()
        
        probs = (1 / (1 + np.exp(-logits))).tolist()
        all_scores.extend(probs)

    return RerankResult(RERANKING_MODEL, documents, all_scores)


def rerank_pinecone(pc: Pinecone, query: str, documents: list[dict], top_n: int = RERANK_TOP_N) -> RerankResult:
    result = pc.inference.rerank(
        model="pinecone-rerank-v0",
        query=query,
        documents=documents,
        top_n=top_n,
        rank_fields=["chunk_text"],
        return_documents=True,
        parameters={"truncate": "END"}
    )
    scores = [d["score"] for d in result.data]
    docs = [d["document"] for d in result.data]
    return RerankResult("pinecone-rerank-v0", docs, scores)