from app.services.rag_service import RAGService


def main():
    print("=" * 70)
    print("END-TO-END RAG EVALUATION")
    print("=" * 70)

    rag_service = RAGService()

    test_cases = [
        # --------------------------------------------------
        # DIRECT RELEVANT
        # --------------------------------------------------
        {
            "id": 1,
            "category": "DIRECT RELEVANT",
            "question": "How many unused PTO days can an employee roll over?",
            "expected": "answer",
        },
        {
            "id": 2,
            "category": "DIRECT RELEVANT",
            "question": "What does a flashing amber LED indicate?",
            "expected": "answer",
        },
        {
            "id": 3,
            "category": "DIRECT RELEVANT",
            "question": "What is the maximum throughput of the Edge Router?",
            "expected": "answer",
        },
        {
            "id": 4,
            "category": "DIRECT RELEVANT",
            "question": "What was the adjusted EBITDA for FY2025?",
            "expected": "answer",
        },
        {
            "id": 5,
            "category": "DIRECT RELEVANT",
            "question": "How long can EU customer PII remain in production databases?",
            "expected": "answer",
        },

        # --------------------------------------------------
        # SEMANTICALLY RELEVANT
        # --------------------------------------------------
        {
            "id": 6,
            "category": "SEMANTIC RELEVANT",
            "question": "If an employee does not use all their leave, how many days can they carry forward?",
            "expected": "answer",
        },
        {
            "id": 7,
            "category": "SEMANTIC RELEVANT",
            "question": "What condition causes the router's amber status light to flash?",
            "expected": "answer",
        },
        {
            "id": 8,
            "category": "SEMANTIC RELEVANT",
            "question": "What is the Edge Router's maximum firewall processing capacity?",
            "expected": "answer",
        },

        # --------------------------------------------------
        # IRRELEVANT
        # --------------------------------------------------
        {
            "id": 9,
            "category": "IRRELEVANT",
            "question": "Who is the Prime Minister of India?",
            "expected": "reject",
        },
        {
            "id": 10,
            "category": "IRRELEVANT",
            "question": "What is the capital of France?",
            "expected": "reject",
        },
        {
            "id": 11,
            "category": "IRRELEVANT",
            "question": "What is the company's maternity leave policy?",
            "expected": "reject",
        },

        # --------------------------------------------------
        # HALLUCINATION / UNSUPPORTED
        # --------------------------------------------------
        {
            "id": 12,
            "category": "UNSUPPORTED",
            "question": (
                "What is the company's maternity leave policy "
                "for employees with more than 10 years of service?"
            ),
            "expected": "reject",
        },
    ]

    passed = 0
    failed = 0

    for test in test_cases:
        print("\n" + "=" * 70)
        print(f"TEST {test['id']}")
        print("=" * 70)

        print(f"Category : {test['category']}")
        print(f"Question : {test['question']}")
        print(f"Expected : {test['expected']}")

        try:
            answer = rag_service.answer(
                question=test["question"]
            )

            print("\nANSWER")
            print("-" * 70)
            print(answer)

            # --------------------------------------------------
            # Basic behavioral validation
            # --------------------------------------------------
            rejection_phrases = [
                "i could not find the answer",
                "could not find the answer",
                "not found in the provided documents",
                "not available in the provided documents",
            ]

            is_rejected = any(
                phrase in answer.lower()
                for phrase in rejection_phrases
            )

            if test["expected"] == "answer":
                if not is_rejected:
                    print("\nRESULT : PASS")
                    passed += 1
                else:
                    print("\nRESULT : FAIL")
                    print("Expected an answer, but RAG rejected the question.")
                    failed += 1

            elif test["expected"] == "reject":
                if is_rejected:
                    print("\nRESULT : PASS")
                    passed += 1
                else:
                    print("\nRESULT : FAIL")
                    print("Expected rejection, but system generated an answer.")
                    failed += 1

        except Exception as exc:
            print("\nRESULT : FAIL")
            print(f"ERROR  : {type(exc).__name__}: {exc}")
            failed += 1

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total = passed + failed

    print(f"Total Tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")

    if total:
        accuracy = (passed / total) * 100
        print(f"Pass Rate   : {accuracy:.2f}%")

    print("=" * 70)
    print("END-TO-END RAG EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()