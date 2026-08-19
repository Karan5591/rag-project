from collections import Counter


def calculate_metrics(results: list[dict]) -> dict:
    """
    Calculate evaluation metrics from individual test results.
    """

    total = len(results)

    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
        }

    passed = sum(1 for result in results if result["passed"])
    failed = total - passed

    category_counts = Counter(
        result["category"]
        for result in results
    )

    category_passed = Counter(
        result["category"]
        for result in results
        if result["passed"]
    )

    category_metrics = {}

    for category, count in category_counts.items():
        category_metrics[category] = {
            "total": count,
            "passed": category_passed[category],
            "pass_rate": (
                category_passed[category] / count * 100
            ),
        }

    relevant_cases = [
        result
        for result in results
        if result["expected"] == "answer"
    ]

    rejection_cases = [
        result
        for result in results
        if result["expected"] == "reject"
    ]

    relevant_passed = sum(
        1
        for result in relevant_cases
        if result["passed"]
    )

    rejection_passed = sum(
        1
        for result in rejection_cases
        if result["passed"]
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total * 100,
        "relevant_total": len(relevant_cases),
        "relevant_passed": relevant_passed,
        "relevant_accuracy": (
            relevant_passed / len(relevant_cases) * 100
            if relevant_cases
            else 0.0
        ),
        "rejection_total": len(rejection_cases),
        "rejection_passed": rejection_passed,
        "rejection_accuracy": (
            rejection_passed / len(rejection_cases) * 100
            if rejection_cases
            else 0.0
        ),
        "category_metrics": category_metrics,
    }