import gc
import json
import traceback
from tqdm import tqdm
from pathlib import Path

from src.utils.logger import logger
from src.ingestion.config import PINECONE_NAMESPACE as NAMESPACE
from src.ingestion.vector_store import get_pinecone_index
from src.ingestion.text_processor import convert_to_text, split_text
from src.ingestion.embedder import (
    load_dense_embedding_model,
    load_sparse_embedding_model,
    get_dense_embedding,
    get_sparse_embedding,
)


def extract_necessary_data(json_data):
    embedding_keys = [json_data['title_field_name']] + json_data['content_field_names']
    metadata_keys = [k for k in json_data.keys() if k not in embedding_keys]

    participant_fields = [
        'author', 'reviewers', 'redwoood_owner', 'redwood_attendees',
        'mailbox_owner', 'participants_internal', 'owner',
        'collaborators', 'reporter', 'assignee', 'creator', 'participants'
    ]

    def sanitize_field(val):
        if isinstance(val, str):
            return [val]
        elif isinstance(val, list):
            return val
        return []

    participants_keys = set()
    for field in participant_fields:
        participants_keys.update(sanitize_field(json_data.get(field)))

    return {
        'embedding_keys': embedding_keys,
        'metadata_keys': metadata_keys,
        'hybrid_keys': [],
        'user_access': list(participants_keys)
    }


def _get_source(filepath: Path) -> str:
    parts = filepath.parts
    return parts[7] if len(parts) > 7 else "unknown"


def process_file_to_pinecone(filepath, dense_tokenizer, dense_model,
                              sparse_tokenizer, sparse_session, sparse_input_names, index):
    """Process a JSON file and return vectors for Pinecone upsert."""
    try:
        source = _get_source(filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        clean_dict = extract_necessary_data(json_data)
        embedding_keys = clean_dict.get("embedding_keys", [])
        hybrid_keys = clean_dict.get("hybrid_keys", [])
        user_access = clean_dict.get('user_access', [])

        processed_json = {k: v for k, v in json_data.items() if k in set(embedding_keys + hybrid_keys)}
        key_value_text = convert_to_text(processed_json)
        chunks = split_text(key_value_text)

        vectors = []
        for idx, chunk in enumerate(chunks):
            vector_id = f"{source}_{filepath.stem}_chunk_{idx:04d}"

            dense_embedding = get_dense_embedding(dense_tokenizer, dense_model, chunk.page_content)[0].tolist()
            sparse_indices, sparse_values = get_sparse_embedding(
                sparse_tokenizer, sparse_session, sparse_input_names, chunk.page_content
            )

            chunk_metadata = {
                "source": str(filepath.parts[7]),
                "source_path": '/'.join(filepath.parts[7:]),
                "file_name": filepath.name,
                "chunk_index": idx,
                "chunk_size": len(chunk.page_content),
                "text": chunk.page_content,
                "user_access": user_access,
                'dataset_doc_uuid': json_data['dataset_doc_uuid']
            }

            vectors.append({
                "id": vector_id,
                "values": dense_embedding,
                "sparse_values": {
                    "indices": sparse_indices,
                    "values": sparse_values
                },
                "metadata": chunk_metadata
            })

        return vectors

    except Exception as e:
        print(f"Process file error ({filepath}): {traceback.format_exc()}")
        raise


def process_all_files(data_dir, dense_tokenizer, dense_model,
                      sparse_tokenizer, sparse_session, sparse_input_names, index):
    """Process all JSON files in data_dir sequentially with batched upserts."""
    json_files = list(Path(data_dir).rglob("*.json"))
    print(f"Found {len(json_files)} JSON files to process")

    results = []
    errors = []

    for idx, filepath in tqdm(enumerate(json_files)):
        try:
            vectors = process_file_to_pinecone(
                filepath, dense_tokenizer, dense_model,
                sparse_tokenizer, sparse_session, sparse_input_names, index
            )
            results.extend(vectors)

            if len(results) >= 25:
                print('Upserting vectors to db')
                index.upsert(vectors=results, namespace=NAMESPACE)
                results = []

        except Exception as e:
            traceback.print_exc()
            errors.append({"file": str(filepath), "error": str(e)})
            print(f"[{idx + 1}/{len(json_files)}] Error: {filepath}: {e}")
            break

    return {"results": results, "errors": errors}


if __name__ == "__main__":
    data_dir = './data'

    dense_tokenizer, dense_model = load_dense_embedding_model()
    sparse_tokenizer, sparse_session, sparse_input_names = load_sparse_embedding_model()
    _, index = get_pinecone_index()

    summary = process_all_files(
        data_dir, dense_tokenizer, dense_model,
        sparse_tokenizer, sparse_session, sparse_input_names, index
    )

    logger.info(f"\nProcessed files successfully, {len(summary['results'])} vectors pending upsert")

    gc.collect()

    if summary["errors"]:
        logger.error(f"Errors: {len(summary['errors'])}")
        for err in summary["errors"]:
            logger.error(f"  - {err['file']}: {err['error']}")
