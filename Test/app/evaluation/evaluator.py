from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.metrics import calculate_metrics
from app.services.retrieval_service import RetrievalService


class RAGEvaluator:
    """
    Runs the RAG evaluation dataset against the
    retrieval pipeline.
    """

    def __init__(self):
        self.retrieval_service = RetrievalService()

    def evaluate(self) -> dict:
        results = []

        for test_case in EVALUATION_DATASET:
            question = test_case["question"]
            expected_keywords = test_case.get("expected_keywords")

            retrieved = self.retrieval_service.retrieve(question, top_k=3)

            was_retrieved = len(retrieved) > 0

            # ------------------------------------------------------
            # If the dataset gives us expected_keywords, "passed" means
            # the retrieved chunk content actually CONTAINS those
            # keywords — not just that something scored above the
            # relevance threshold. This is what separates "the right
            # answer came back" from "some plausible-looking chunk
            # came back" (the gap the earlier retrieval-only run
            # couldn't see, e.g. T003/T008's low-margin scores).
            # Falls back to presence/absence for cases without
            # expected_keywords, same as before.
            # ------------------------------------------------------
            missing_keywords = []

            if expected_keywords:
                combined_content = " ".join(
                    doc.get("content", "") for doc in retrieved
                ).lower()

                missing_keywords = [
                    kw for kw in expected_keywords
                    if kw.lower() not in combined_content
                ]

                passed = was_retrieved and not missing_keywords

            elif test_case["expected"] == "answer":
                passed = was_retrieved
            else:
                passed = not was_retrieved

            results.append(
                {
                    "id": test_case["id"],
                    "category": test_case["category"],
                    "question": question,
                    "expected": test_case["expected"],
                    "retrieved": was_retrieved,
                    "retrieved_count": len(retrieved),
                    "missing_keywords": missing_keywords,
                    "passed": passed,
                    "top_score": (
                        retrieved[0]["reranker_score"]
                        if retrieved
                        else None
                    ),
                }
            )

        metrics = calculate_metrics(results)

        return {
            "results": results,
            "metrics": metrics,
        }