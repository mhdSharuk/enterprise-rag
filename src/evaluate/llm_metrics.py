import re
import json
import numpy as np
from tqdm import tqdm
from pinecone import data
from pinecone import data
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (ContextualRecallMetric,
                              ContextualRelevancyMetric,
                              AnswerRelevancyMetric,
                              FaithfulnessMetric,
                              GEval)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src.evaluate.config import GroqLLM, TEST_SAMPLE

groq_model = GroqLLM()
np.random.seed(69)

context_relevancy  = ContextualRelevancyMetric(model=groq_model)
contextual_recall  = ContextualRecallMetric(model=groq_model)
answer_relevancy   = AnswerRelevancyMetric(model=groq_model)
faithufullness     = FaithfulnessMetric(model=groq_model)
answer_correctness = GEval(name="Answer Correctness", model=groq_model,
                            evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, 
                                                SingleTurnParams.ACTUAL_OUTPUT],
                            evaluation_steps=["""Determine whether the actual output is 
                                                factually correct based on the expected output."""]
)

offline_metrics_list = [
    contextual_recall,
    answer_correctness,
    faithufullness,
]

online_metrics_list = [
  context_relevancy,
  answer_relevancy,
  faithufullness,
]

test_cases = []

def read_response_data():
    
    with open('src/evaluate/results/final_response_data.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]

    return data

def get_sample_test_cases(data):
    selected_test_cases = np.random.choice([x for x in data], TEST_SAMPLE, replace=False)

    test_cases = []
    test_cases.append(LLMTestCase(
        input = selected_test_cases[i]['question_data']['question'],
        actual_output = re.sub(r'<tool_call>.*?<tool_call>', '',
                                selected_test_cases[i]['answer_data']['answer'],
                                flags=re.DOTALL).strip(),
        expected_output = data[i]['question_data']['gold_answer'],
        retrieval_context=['.'.join(resp['text'].strip().replace('\\n', '\n').split('\n')) 
                           for resp in selected_test_cases[i]['response_data']]
    ))

    return test_cases

def evaluate_offline_test_cases(test_cases, contextual_recall, 
                                answer_correctness, faithufullness):
    
    contextual_recall_score = []
    answer_correctness_score = []
    faithufullness_score = []

    contextual_recall_reason = []
    answer_correctness_reason = []
    faithufullness_reason = []

    for test_case in tqdm(test_cases):

        contextual_recall.measure(test_case)
        contextual_recall_score.append(contextual_recall.score)
        contextual_recall_reason.append(contextual_recall.reason)
        
        answer_correctness.measure(test_case)
        answer_correctness_score.append(answer_correctness.score)
        answer_correctness_reason.append(answer_correctness.reason)

        faithufullness.measure(test_case)
        faithufullness_score.append(faithufullness.score)
        faithufullness_reason.append(faithufullness.reason)

    avg_contextual_recall = np.mean(contextual_recall_score)
    avg_answer_correctness = np.mean(answer_correctness_score)
    avg_faithufullness = np.mean(faithufullness_score)

    result = {
        'avg_contextual_recall': avg_contextual_recall,
        'avg_answer_correctness': avg_answer_correctness,
        'avg_faithufullness': avg_faithufullness
    }

    with open('src/evaluate/results/llm_offline_evaluation_results.json', 'w') as f:
        json.dump(result, f, indent=4)


def evaluate_online_test_cases(query, response, context, context_relevancy, answer_relevancy, faithufullness):
    test_case = LLMTestCase(input=query, actual_output=response, retrieval_context=context)
    context_relevancy.measure(test_case)
    answer_relevancy.measure(test_case)
    faithufullness.measure(test_case)

    scores = {
        'context_relevancy_score': context_relevancy.score,
        'answer_relevancy_score': answer_relevancy.score,
        'faithufullness_score': faithufullness.score,
    }
    
    return scores

data = read_response_data()
test_cases = get_sample_test_cases(data)
evaluate_offline_test_cases(test_cases, contextual_recall, answer_correctness, faithufullness)