import os
import gc
import sys
import json
import torch
import traceback
from tqdm import tqdm
from pathlib import Path

from src.utils.logger import logger
from src.ingestion.vector_store import upsert_vectors
from src.ingestion.embedder import load_embedding_model
from src.ingestion.vector_store import get_pinecone_index
from src.ingestion.key_extractor import extract_keys_from_schema
from src.ingestion.text_processor import convert_to_text, split_text
from src.ingestion.embedder import get_dense_embedding, get_sparse_embedding

def _get_source(filepath: Path) -> str:
    parts = filepath.parts
    return parts[7] if len(parts) > 7 else "unknown"


def process_file(filepath: Path, embedding_model, pc, index) -> dict:
    try:
        source = _get_source(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        key_schema = extract_keys_from_schema(json_data)
        embedding_keys = key_schema.get("embedding_keys", [])
        metadata_keys = key_schema.get("metadata_keys", [])
        hybrid_keys = key_schema.get("hybrid_keys", [])

        content_json = {k: v for k, v in json_data.items() if k in set(embedding_keys + hybrid_keys)}
        text = convert_to_text(content_json)
        chunks = split_text(text)

        vectors = []
        for idx, chunk in enumerate(chunks):
            vector_id = f"{source}_{filepath.stem}_chunk_{idx:04d}"

            dense = get_dense_embedding(embedding_model, chunk.page_content)            
            sparse_indices, sparse_values = get_sparse_embedding(pc, chunk.page_content)

            metadata = {
                "source": source,
                "source_path": "/".join(filepath.parts[7:]),
                "file_name": filepath.name,
                "chunk_index": idx,
                "chunk_size": len(chunk.page_content),
                "text": chunk.page_content,
            }

            for key in set(metadata_keys + hybrid_keys):
                if key in json_data:
                    metadata[key] = json_data[key]

            vectors.append({
                "id": vector_id,
                "values": dense,
                "sparse_values": {
                    "indices": sparse_indices,
                    "values": sparse_values
                },
                "metadata": metadata
            })

        upsert_vectors(index, vectors)

        return {
            "input_file": str(filepath),
            "chunks": len(chunks),
            "embedding_keys": embedding_keys,
            "metadata_keys": metadata_keys
        }

    except Exception:
        logger.error(f"process_file error ({filepath}): {traceback.format_exc()}")
        raise


def process_all_files(data_dir: str, embedding_model, pc, index) -> dict:
    json_files = list(Path(data_dir).rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {data_dir}")

    results = []
    errors = []

    for idx, filepath in tqdm(enumerate(json_files), total=len(json_files)):
        try:
            result = process_file(filepath, embedding_model, pc, index)
            results.append(result)
        except Exception as e:
            errors.append({"file": str(filepath), "error": str(e)})
            print(f"[{idx + 1}/{len(json_files)}] Error: {filepath}: {e}")

    return {"results": results, "errors": errors}


if __name__ == "__main__":
    data_dir = './data'

    embedding_model = load_embedding_model()
    pc, index = get_pinecone_index()

    summary = process_all_files(data_dir, embedding_model, pc, index)

    logger.info(f"\nProcessed {len(summary['results'])} files successfully")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if summary["errors"]:
        logger.error(f"Errors: {len(summary['errors'])}")
        for err in summary["errors"]:
            logger.error(f"  - {err['file']}: {err['error']}")