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
RERANKING_MODEL = os.getenv("RERANKING_MODEL")
RERANKING_ONNX_FILE = os.getenv('RERANKING_ONNX_FILE')
CHAT_MODEL      = os.getenv("QMODEL_ID", "Qwen/Qwen2.5-Coder-1.5B-Instruct")

SOURCES = ["confluence", "fireflies", "github", "gmail", "google_drive", "hubspot", "jira", "linear", "slack"]

TOP_K_PER_SOURCE = 5
RERANK_TOP_N     = 10
HYBRID_ALPHA     = 0.75
LOCAL_RERANK     = True
RERANK_THRESHOLD = 0.1