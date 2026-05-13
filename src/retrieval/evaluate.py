import os
import json
from tqdm import tqdm
from pprint import pprint

from src.retrieval.query import retrieve
from src.retrieval.embedder import load_embedding_model
from src.retrieval.vector_store import get_pinecone_index
from src.retrieval.metrics import compute_metrics, aggregate_metrics


def load_eval_set(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def run_evaluation(eval_path: str, results_output_path: str | None = None):
    embedding_model = load_embedding_model()
    pc, index = get_pinecone_index()

    questions = load_eval_set(eval_path)
    print(f"Loaded {len(questions)} questions from {eval_path}")

    per_question_metrics = []
    detailed_results = []

    for q in tqdm(questions, desc="Evaluating"):
        question_id = q["question_id"]
        query = q["question"]
        expected_doc_ids = q["expected_doc_ids"]

        merged_docs = retrieve(query, embedding_model, pc, index)

        metrics = compute_metrics(question_id, merged_docs, expected_doc_ids)
        per_question_metrics.append(metrics)

        detailed_results.append({
            "question_id": question_id,
            "question": query,
            "expected_doc_ids": list(expected_doc_ids.keys()),
            "retrieved_doc_ids": [d["id"] for d in merged_docs],
            "hit": metrics.hit,
            "precision_at_k": metrics.precision_at_k,
            "recall_at_k": metrics.recall_at_k,
            "mrr": metrics.mrr,
            "ndcg_at_k": metrics.ndcg_at_k,
            "average_precision": metrics.average_precision,
        })

    summary = aggregate_metrics(per_question_metrics)

    print("\n=== Evaluation Summary ===")
    pprint(summary)

    if results_output_path:
        output = {
            "summary": summary,
            "per_question": detailed_results
        }
        with open(results_output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nDetailed results saved to {results_output_path}")

    return summary, detailed_results


if __name__ == "__main__":
    eval_file = sys.argv[1] if len(sys.argv) > 1 else "questions_a.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "eval_results.json"

    run_evaluation(eval_file, output_file)