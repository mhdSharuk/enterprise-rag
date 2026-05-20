from src.retrieval.config import (PINECONE_API_KEY, PINECONE_INDEX_NAME, 
                                  SOURCES, TOP_K_PER_SOURCE, PINECONE_NAMESPACE)
from src.retrieval.reranker import RerankResult

def get_dense_embeddings_batch(tokenizer, model, texts) -> list[list[float]]:
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        return np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(input_mask_expanded.sum(1), 1e-9)

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )

    outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs["attention_mask"])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)

    if isinstance(texts, str):
      return embeddings[0].tolist()
    elif isinstance(texts, list):
      return embeddings.tolist()

def get_dense_embeddings_batch_chunked(tokenizer, model, texts: list[str],
                                       chunk_size: int = 25) -> list[list[float]]:
    all_embeddings = []
    for i in tqdm(range(0, len(texts), chunk_size)):
        chunk = texts[i: i + chunk_size]
        all_embeddings.extend(get_dense_embeddings_batch(tokenizer, model, chunk))
    return all_embeddings

def rerank_local_batch(
    tokenizer, 
    model, 
    query_docs_batch: list[dict], 
    batch_size: int = 4) -> list[dict]:
    
    if not query_docs_batch:
        return []
    
    inference_queue = []

    for query_idx, item in enumerate(query_docs_batch):
        query = item["question_data"]["question"]
        documents = item["response_data"]

        for doc in documents:
            inference_queue.append((query_idx, query, doc))
    
    if not inference_queue:
        return [
            {
                "question_data": item["question_data"],
                "response_data": []
            }
            for item in query_docs_batch
        ]
    
    results_buffer = {i: [] for i in range(len(query_docs_batch))}
    
    for i in range(0, len(inference_queue), batch_size):
        batch_items = inference_queue[i : i + batch_size]
        
        queries = [item[1] for item in batch_items]
        doc_texts = [item[2]["chunk_text"] for item in batch_items]
        
        inputs = tokenizer(
            queries,
            doc_texts,
            padding=True,
            truncation=True,
            return_tensors="np" 
        )
        
        outputs = model(**inputs)
        logits = outputs.logits.flatten()
        probs = (1 / (1 + np.exp(-logits))).tolist()
        
        for idx, (query_idx, _, doc) in enumerate(batch_items):
            results_buffer[query_idx].append(
                (doc, float(logits[idx]), float(probs[idx]))
            )
    
    final_results = []

    for query_idx, item in enumerate(query_docs_batch):
        question_data = item["question_data"]

        if results_buffer[query_idx]:
            docs_list = [x[0] for x in results_buffer[query_idx]]
            logits_list = [x[1] for x in results_buffer[query_idx]]
            scores_list = [x[2] for x in results_buffer[query_idx]]

            reranked = RerankResult(
                RERANKING_MODEL,
                docs_list,
                scores_list,
                logits_list
            )

            final_results.append({
                "question_data": question_data,
                "response_data": reranked.data
            })
        else:
            final_results.append({
                "question_data": question_data,
                "response_data": []
            })
    
    return final_results

def query_all_sources_batch(index, batch: list[dict], top_k=TOP_K_PER_SOURCE) -> list[dict]:
    query_futures = {}

    for item in batch:
        question_id = item["question_id"]
        query_futures[question_id] = {}
        for source in SOURCES:
            future = index.query(
                top_k=top_k,
                vector=item["dense_vector"],
                # sparse_vector=sparse_vector if sparse_vector and sparse_vector["indices"] else None,
                include_values=False,
                include_metadata=True,
                filter={"source": {"$eq": source}},
                async_req=True,
                namespace=PINECONE_NAMESPACE
            )
            query_futures[question_id][source] = future

    results = []
    batch_map = {item["question_id"]: item for item in batch}

    for question_id, source_futures in query_futures.items():
        documents = []
        for source, future in source_futures.items():
            response = future.get()
            matches = response.to_dict()["matches"]
            docs = [
                {
                    "id": m["id"],
                    "score": m["score"],
                    "chunk_text": m["metadata"]["text"],
                    "doc_id": m["metadata"]["dataset_doc_uuid"]
                }
                for m in matches
            ]
            documents.extend(docs)

        item = batch_map[question_id]
        results.append({
            "question_data": {
                "question_id": item["question_id"],
                "question": item["question"],
                "expected_doc_ids": item["expected_doc_ids"],
                "gold_answer": item["gold_answer"],
                "answer_facts": item["answer_facts"],
            },
            "response_data": documents
        })

    return results

def run_batch_queries(queries: list[dict], pc, pc_index,
                      dense_tokenizer, dense_model, 
                      reranker_tokenizer, reranker_model,
                      batch_size: int = 25) -> list[dict]:
    all_results = []

    for batch_start in range(0, len(queries), batch_size):
        batch_queries = queries[batch_start: batch_start + batch_size]
        query_texts = [item["question"] for item in batch_queries]

        dense_embeddings = get_dense_embeddings_batch_chunked(
            dense_tokenizer, dense_model, query_texts
        )

        batch = [
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "expected_doc_ids": item["expected_doc_ids"],
                "gold_answer": item["gold_answer"],
                "answer_facts": item["answer_facts"],
                "dense_vector": dense_emb,
                # "sparse_vector": sparse_emb
            }
            for item, dense_emb in zip(batch_queries, dense_embeddings)
        ]

        batch_results = query_all_sources_batch(pc_index, batch)

        final_results = rerank_local_batch(reranker_tokenizer, 
                                           reranker_model, batch_results, batch_size=1)

        all_results.extend(final_results)

    return all_results