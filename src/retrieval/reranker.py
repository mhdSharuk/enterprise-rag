# from pathlib import Path
# from pinecone import Pinecone
# from sentence_transformers import CrossEncoder

# from src.retrieval.config import RERANKING_MODEL, HF_TOKEN, RERANK_TOP_N

# MODEL_STORE = Path("model_store")
# local_reranker_model = None

# class RerankResult:
#     def __init__(self, model_name: str, documents: list, scores: list):
#         self.model = model_name
#         self.data = sorted(
#             [{"score": float(s), "chunk_text": d} for s, d in zip(scores, documents)],
#             key=lambda x: x["score"],
#             reverse=True
#         )

#     def __repr__(self):
#         return f"RerankResult(model='{self.model}', data={self.data})"

# def load_local_reranker():
#     global local_reranker_model

#     if local_reranker_model is None:
#         model_name = RERANKING_MODEL.replace("/", "_")
#         local_path = MODEL_STORE / model_name

#         if local_path.exists():
#             local_reranker_model = CrossEncoder(str(local_path))
#         else:
#             MODEL_STORE.mkdir(parents=True, exist_ok=True)
#             local_reranker_model = CrossEncoder(RERANKING_MODEL, token=HF_TOKEN)
#             local_reranker_model.save(str(local_path))

# def rerank_local(query: str, documents: list[dict], batch_size: int = 30) -> RerankResult:
#     load_local_reranker()

#     pairs = [[query, doc.get("chunk_text", doc.get("text", ""))] for doc in documents]
#     scores = local_reranker_model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
#     return RerankResult(RERANKING_MODEL, documents, scores)

# def rerank_pinecone(pc: Pinecone, query: str, documents: list[dict], top_n: int = RERANK_TOP_N) -> RerankResult:
#     result = pc.inference.rerank(
#         model="pinecone-rerank-v0",
#         query=query,
#         documents=documents,
#         top_n=top_n,
#         rank_fields=["chunk_text"],
#         return_documents=True,
#         parameters={"truncate": "END"}
#     )
#     scores = [d["score"] for d in result.data]
#     docs = [d["document"] for d in result.data]
#     return RerankResult("pinecone-rerank-v0", docs, scores)


from pathlib import Path
import numpy as np
from pinecone import Pinecone
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, logging as transformers_logging
from src.retrieval.config import RERANKING_MODEL, HF_TOKEN, RERANK_TOP_N

from optimum.utils import logging as optimum_logging

MODEL_STORE = Path("model_store")
RERANKER_PATH = MODEL_STORE / RERANKING_MODEL.replace("/", "_")

transformers_logging.set_verbosity_info()
optimum_logging.set_verbosity_info()

def load_reranker():
    if RERANKER_PATH.exists():
        print(f"Loading reranker from {RERANKER_PATH}")

        tokenizer = AutoTokenizer.from_pretrained(str(RERANKER_PATH))

        local_file = "model_quantized.onnx" if (RERANKER_PATH / "model_quantized.onnx").exists() else "model.onnx"
        model = ORTModelForSequenceClassification.from_pretrained(
            str(RERANKER_PATH),
            file_name=local_file,
            token=HF_TOKEN
        )
        return tokenizer, model

    print(f"Exporting reranker {RERANKING_MODEL} to ONNX ...")
    MODEL_STORE.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(RERANKING_MODEL, token=HF_TOKEN)
    model = ORTModelForSequenceClassification.from_pretrained(
        RERANKING_MODEL,
        subfolder="onnx",
        # file_name="model_quantized.onnx",
        file_name = 'model.onnx',
        token=HF_TOKEN
    )

    model.save_pretrained(str(RERANKER_PATH))
    tokenizer.save_pretrained(str(RERANKER_PATH))
    print(f"Saved ONNX reranker to {RERANKER_PATH}")

    return tokenizer, model


class RerankResult:
    def __init__(self, model_name: str, documents: list, scores: list):
        self.model = model_name
        self.data = sorted(
            [{"score": float(s), "document": d} for s, d in zip(scores, documents)],
            key=lambda x: x["score"],
            reverse=True
        )

    def __repr__(self):
        return f"RerankResult(model='{self.model}', data={self.data})"


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