"""
End-to-end test for the four buried-detail failure cases documented in
Initial_Testing_Report.docx.

Unlike test_evaluation_framework.py / test_rag_evaluation.py, this runs
the FULL pipeline via RAGService.answer() — retrieval, reranking,
context building, prompt construction, AND the Llama call — and checks
the actual generated answer text. This is the only one of the three
harnesses that can catch a failure introduced after retrieval (e.g. in
context_builder's filtering, prompt construction, or generation itself).

Run with:
    python -m app.test_rag_hard_cases
"""

from app.services.rag_service import RAGService


HARD_CASES = [
    {
        "id": "T013",
        "label": "PTO rollover exception (deadline + approver)",
        "question": (
            "What is the deadline for obtaining a written exception to "
            "the PTO rollover forfeiture rule, and who must approve it?"
        ),
        "expected_keywords": ["December 15", "VP of HR"],
    },
    {
        "id": "T014",
        "label": "Tier 2 international travel (flight class + hotel cap)",
        "question": (
            "I'm a Tier 2 employee traveling to Europe for a meeting. "
            "What flight class and hotel budget am I allowed?"
        ),
        "expected_keywords": ["Business Class", "$500"],
    },
    {
        "id": "T015",
        "label": "Highest-growth region (region + revenue, from table)",
        "question": (
            "Which region had the highest YoY growth rate in FY2025, "
            "and what was its total revenue?"
        ),
        "expected_keywords": ["APAC", "22.5%", "74.6"],
    },
    {
        "id": "T016",
        "label": "Router thermal threshold",
        "question": (
            "At what internal core temperature does the Edge Router "
            "trigger a fan failure warning?"
        ),
        "expected_keywords": ["75"],
    },
]


def main():
    print("=" * 70)
    print("HARD-CASE END-TO-END RAG TEST (retrieval + rerank + Llama)")
    print("=" * 70)

    rag_service = RAGService()

    passed = 0
    failed = 0

    for case in HARD_CASES:
        print()
        print("=" * 70)
        print(f"{case['id']} | {case['label']}")
        print("=" * 70)
        print(f"Question : {case['question']}")

        try:
            answer = rag_service.answer(question=case["question"])
        except Exception as exc:
            print(f"\nRESULT : FAIL (exception)")
            print(f"ERROR  : {type(exc).__name__}: {exc}")
            failed += 1
            continue

        print("\nANSWER")
        print("-" * 70)
        print(answer)

        answer_lower = answer.lower()
        missing = [
            kw for kw in case["expected_keywords"]
            if kw.lower() not in answer_lower
        ]

        if not missing:
            print("\nRESULT : PASS")
            passed += 1
        else:
            print(f"\nRESULT : FAIL")
            print(f"Missing from final answer: {missing}")
            print(
                "(If retrieval/reranking DID find this detail but it's "
                "still missing here, the bug is downstream of retrieval — "
                "check context_builder filtering, prompt construction, "
                "or generation, not the retriever.)"
            )
            failed += 1

    total = passed + failed

    print()
    print("=" * 70)
    print("HARD-CASE SUMMARY")
    print("=" * 70)
    print(f"Total  : {total}")
    print(f"Passed : {passed}")
    print(f"Failed : {failed}")
    if total:
        print(f"Rate   : {(passed / total) * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
