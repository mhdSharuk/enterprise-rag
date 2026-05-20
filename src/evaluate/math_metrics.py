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