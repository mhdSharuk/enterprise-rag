import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(os.getcwd()) / ".env"
load_dotenv(dotenv_path=env_path, override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

HF_TOKEN = os.getenv("HF_TOKEN")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_ONNX_FILE = os.getenv('EMBEDDING_ONNX_FILE')

CACHE_SEMANTIC_TOP_K = 5
CACHE_SEMANTIC_THRESHOLD = 0.5
CACHE_FINAL_THRESHOLD = 0.5

HYBRID_SCORE_ALPHA = 0.5
BM25_SIGMOID_SCALE = 1.0