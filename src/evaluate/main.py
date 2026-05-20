import json
from src.evaluate.math_metrics import recall_at_k, mean_reciprocal_rank
from src.generation.main import initialize_search_pipeline 
from src.evaluate.utils import run_batch_queries

pc, pc_index, dense_tokenizer, dense_model, reranker = initialize_search_pipeline()

questions = []
with open('questions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

all_results = run_batch_queries(questions, pc, pc_index,
                  dense_tokenizer, dense_model,
                  reranker_tokenizer, reranker_model,
                  batch_size = 25)

recall_5 = recall_at_k(all_results, 5)
recall_10 = recall_at_k(all_results, 10)
mrr_5 = mean_reciprocal_rank(all_results)