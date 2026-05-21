import json
import numpy as np
from tqdm import tqdm
from deepeval.metrics import (ContextualRecallMetric)
from deepeval.test_case import LLMTestCase

from src.utils.logger import logger
from src.evaluate.config import GroqLLM, langfuse, TEST_SAMPLE
from src.generation.main import initialize_search_pipeline 
from src.evaluate.utils import run_batch_queries

def recall_at_k(results: list[dict], k: int) -> float:
    if not results:
        return 0.0

    hits = 0

    for item in results:
        expected_doc_ids = item["question_data"]["expected_doc_ids"]

        if isinstance(expected_doc_ids, str):
            expected_doc_ids = {expected_doc_ids}
        else:
            expected_doc_ids = set(expected_doc_ids)

        seen = set()
        ranked_doc_ids = []

        for doc in item["response_data"]:
            doc_id = doc["doc_id"]

            if doc_id not in seen:
                seen.add(doc_id)
                ranked_doc_ids.append(doc_id)

        if any(doc_id in expected_doc_ids for doc_id in ranked_doc_ids[:k]):
            hits += 1

    return hits / len(results)

def mean_reciprocal_rank(results: list[dict]) -> float:
    if not results:
        return 0.0

    reciprocal_ranks = []

    for item in results:
        expected_doc_ids = item["question_data"]["expected_doc_ids"]

        if isinstance(expected_doc_ids, str):
            expected_doc_ids = {expected_doc_ids}
        else:
            expected_doc_ids = set(expected_doc_ids)

        seen = set()
        rank = 0

        for idx, doc in enumerate(item["response_data"], start=1):
            doc_id = doc["doc_id"]

            if doc_id in seen:
                continue

            seen.add(doc_id)
            rank += 1

            if doc_id in expected_doc_ids:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(results)

def read_evaluation_data():
    with open('src/evaluate/dataset/questions.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]

    return data

def store_retrieval_response_data(results):
    logger.info("Storing retrieval response data...")
    with open('src/evaluate/results/retrieval_response_data.jsonl', 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

def evaluate_retireval_with_llm(results, contextual_recall):
    contextual_recall_score = []
    contextual_recall_reason = []

    results = np.random.choice(results, TEST_SAMPLE, replace=False)

    for item in tqdm(results):
        test_case = LLMTestCase(
            input = item["question_data"]["question"],
            actual_output = '',
            expected_output = item["question_data"]["gold_answer"],
            retrieval_context=['.'.join(resp['chunk_text'].strip().replace('\\n', '\n').split('\n')) 
                               for resp in item['response_data']]
        )

        contextual_recall.measure(test_case)
        contextual_recall_score.append(contextual_recall.score)
        contextual_recall_reason.append(contextual_recall.reason)

    avg_contextual_recall = np.mean(contextual_recall_score)

    return avg_contextual_recall

pc, pc_index, dense_tokenizer, dense_model, reranker = initialize_search_pipeline()
reranker_tokenizer, reranker_model = reranker

groq_model = GroqLLM()

questions = read_evaluation_data()

logger.info("Running retrieval math evaluation...")
all_results = run_batch_queries(questions, pc, pc_index,
                  dense_tokenizer, dense_model,
                  reranker_tokenizer, reranker_model,
                  batch_size = 25)

recall_5 = recall_at_k(all_results, 5)
recall_10 = recall_at_k(all_results, 10)
mrr = mean_reciprocal_rank(all_results)

store_retrieval_response_data(all_results)

contextual_recall = ContextualRecallMetric(model=groq_model)

logger.info("Running retrieval llm evaluation...")
avg_contextual_recall = evaluate_retireval_with_llm(all_results, contextual_recall)

with langfuse.start_as_current_observation(name="deepeval_evaluation", as_type="span") as span:
    span.score(
        name="avg_contextual_recall",
        value=avg_contextual_recall,
    )
    span.score(
        name="recall@5",
        value=recall_5,
    )
    span.score(
        name="recall@10",
        value=recall_10,
    )
    span.score(
        name="mrr",
        value=mrr,
    )