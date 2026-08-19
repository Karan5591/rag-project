from app.evaluation.evaluator import RAGEvaluator


def main():
    print("=" * 70)
    print("RAG EVALUATION FRAMEWORK")
    print("=" * 70)

    evaluator = RAGEvaluator()

    evaluation = evaluator.evaluate()

    results = evaluation["results"]
    metrics = evaluation["metrics"]

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"

        print()
        print("-" * 70)
        print(f"{result['id']} | {result['category'].upper()}")
        print(f"Question : {result['question']}")
        print(f"Expected : {result['expected']}")
        print(f"Retrieved: {result['retrieved_count']}")

        if result["top_score"] is not None:
            print(f"Top Score: {result['top_score']:.6f}")

        if result.get("missing_keywords"):
            print(f"Missing  : {result['missing_keywords']}")

        print(f"Result   : {status}")

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total Tests        : {metrics['total']}")
    print(f"Passed             : {metrics['passed']}")
    print(f"Failed             : {metrics['failed']}")
    print(f"Pass Rate          : {metrics['pass_rate']:.2f}%")

    print()
    print("RETRIEVAL / RELEVANCE")
    print("-" * 70)

    print(
        f"Relevant Accuracy  : "
        f"{metrics['relevant_accuracy']:.2f}% "
        f"({metrics['relevant_passed']}/"
        f"{metrics['relevant_total']})"
    )

    print(
        f"Rejection Accuracy : "
        f"{metrics['rejection_accuracy']:.2f}% "
        f"({metrics['rejection_passed']}/"
        f"{metrics['rejection_total']})"
    )

    print()
    print("BY CATEGORY")
    print("-" * 70)

    for category, data in metrics["category_metrics"].items():
        print(
            f"{category:20} "
            f"{data['passed']}/{data['total']} "
            f"({data['pass_rate']:.2f}%)"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()