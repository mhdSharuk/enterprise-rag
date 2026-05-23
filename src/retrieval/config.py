import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(os.getcwd()) / ".env"
load_dotenv(dotenv_path=env_path, override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

HF_TOKEN = os.getenv("HF_TOKEN")

DENSE_EMBEDDING_MODEL      = os.getenv("DENSE_EMBEDDING_MODEL")
DENSE_EMBEDDING_ONNX_FILE  = os.getenv('DENSE_EMBEDDING_ONNX_FILE')
SPARSE_EMBEDDING_MODEL     = os.getenv("SPARSE_EMBEDDING_MODEL")
SPARSE_EMBEDDING_ONNX_FILE = os.getenv('SPARSE_EMBEDDING_ONNX_FILE')
RERANKING_MODEL            = os.getenv("RERANKING_MODEL")
RERANKING_ONNX_FILE        = os.getenv('RERANKING_ONNX_FILE')

SOURCES = ["confluence", "fireflies", "github", "gmail", "google_drive", "hubspot", "jira", "linear", "slack"]

TOP_K_PER_SOURCE = 5
RERANK_TOP_N     = 10
HYBRID_ALPHA     = 0.5
LOCAL_RERANK     = True
RERANK_THRESHOLD = 0.5