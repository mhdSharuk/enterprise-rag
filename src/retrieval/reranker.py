from pathlib import Path
from pinecone import Pinecone
from sentence_transformers import CrossEncoder

from src.retrieval.config import RERANKING_MODEL, HF_TOKEN, RERANK_TOP_N

MODEL_STORE = Path("model_store")
local_reranker_model = None

class RerankResult:
    def __init__(self, model_name: str, documents: list, scores: list):
        self.model = model_name
        self.data = sorted(
            [{"score": float(s), "chunk_text": d} for s, d in zip(scores, documents)],
            key=lambda x: x["score"],
            reverse=True
        )

    def __repr__(self):
        return f"RerankResult(model='{self.model}', data={self.data})"

def load_local_reranker():
    global local_reranker_model

    if local_reranker_model is None:
        model_name = RERANKING_MODEL.replace("/", "_")
        local_path = MODEL_STORE / model_name

        if local_path.exists():
            local_reranker_model = CrossEncoder(str(local_path))
        else:
            MODEL_STORE.mkdir(parents=True, exist_ok=True)
            local_reranker_model = CrossEncoder(RERANKING_MODEL, token=HF_TOKEN)
            local_reranker_model.save(str(local_path))

def rerank_local(query: str, documents: list[dict], batch_size: int = 30) -> RerankResult:
    load_local_reranker()

    pairs = [[query, doc.get("chunk_text", doc.get("text", ""))] for doc in documents]
    scores = local_reranker_model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return RerankResult(RERANKING_MODEL, documents, scores)

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