import json
import numpy as np
from tqdm import tqdm
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from src.utils.logger import logger
from src.evaluate.config import GroqLLM, langfuse, TEST_SAMPLE
from src.generation.main import initialize_search_pipeline
from src.evaluate.utils import run_batch_queries


def read_evaluation_data():
    with open('src/evaluate/dataset/questions.jsonl', 'r') as f:
        return [json.loads(line) for line in f]


def store_retrieval_response_data(results):
    logger.info("Storing retrieval response data...")
    with open('src/evaluate/results/retrieval_response_data.jsonl', 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')


def _resolve_expected_ids(expected_doc_ids) -> set:
    if isinstance(expected_doc_ids, str):
        return {expected_doc_ids}
    return set(expected_doc_ids)


def _deduplicated_ranked_ids(response_data) -> list:
    seen = set()
    ranked = []
    for doc in response_data:
        doc_id = doc["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def recall_at_k(results, k) -> float:
    if not results:
        return 0.0

    hits = sum(
        1
        for item in results
        if any(
            doc_id in _resolve_expected_ids(item["question_data"]["expected_doc_ids"])
            for doc_id in _deduplicated_ranked_ids(item["response_data"])[:k]
        )
    )

    return hits / len(results)


def mean_reciprocal_rank(results) -> float:
    if not results:
        return 0.0

    def reciprocal_rank(item):
        expected_ids = _resolve_expected_ids(item["question_data"]["expected_doc_ids"])
        for rank, doc_id in enumerate(_deduplicated_ranked_ids(item["response_data"]), start=1):
            if doc_id in expected_ids:
                return 1 / rank
        return 0.0

    rr_scores = [reciprocal_rank(item) for item in results]
    return sum(rr_scores) / len(results)


def compute_statistical_metrics(results) -> dict:
    return {
        "recall@5":  recall_at_k(results, k=5),
        "recall@10": recall_at_k(results, k=10),
        "mrr":       mean_reciprocal_rank(results),
    }


def evaluate_retrieval_with_llm(results, contextual_recall) -> float:
    sampled = np.random.choice(results, TEST_SAMPLE, replace=False)
    scores = []

    for item in tqdm(sampled):
        test_case = LLMTestCase(
            input=item["question_data"]["question"],
            actual_output='',
            expected_output=item["question_data"]["gold_answer"],
            retrieval_context=[
                '.'.join(resp['chunk_text'].strip().replace('\\n', '\n').split('\n'))
                for resp in item['response_data']
            ]
        )
        contextual_recall.measure(test_case)
        scores.append(contextual_recall.score)

    return float(np.mean(scores))


def log_scores_to_langfuse(metrics):
    with langfuse.start_as_current_observation(name="deepeval_evaluation", as_type="span") as span:
        for name, value in metrics.items():
            span.score(name=name, value=value)


def setup_pipeline():
    pc, pc_index, dense_tokenizer, dense_model, reranker = initialize_search_pipeline()
    reranker_tokenizer, reranker_model = reranker
    groq_model = GroqLLM()
    return pc, pc_index, dense_tokenizer, dense_model, reranker_tokenizer, reranker_model, groq_model


def run_evaluation():
    pc, pc_index, dense_tokenizer, dense_model, reranker_tokenizer, reranker_model, groq_model = setup_pipeline()

    questions = read_evaluation_data()

    logger.info("Running retrieval math evaluation...")
    all_results = run_batch_queries(
        questions, pc, pc_index,
        dense_tokenizer, dense_model,
        reranker_tokenizer, reranker_model,
        batch_size=25
    )

    statistical_metrics = compute_statistical_metrics(all_results)

    store_retrieval_response_data(all_results)

    contextual_recall = ContextualRecallMetric(model=groq_model)

    logger.info("Running retrieval llm evaluation...")
    avg_contextual_recall = evaluate_retrieval_with_llm(all_results, contextual_recall)

    all_metrics = {**statistical_metrics, "avg_contextual_recall": avg_contextual_recall}

    log_scores_to_langfuse(all_metrics)

    return all_metrics


if __name__ == "__main__":
    run_evaluation()
