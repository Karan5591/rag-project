from app.services.rag_service import RAGService


def main():
    print("=" * 70)
    print("END-TO-END RAG SERVICE TEST")
    print("=" * 70)

    rag_service = RAGService()

    test_questions = [
        {
            "type": "RELEVANT",
            "question": "How many unused PTO days can an employee roll over?",
        },
        {
            "type": "RELEVANT",
            "question": "What was the adjusted EBITDA for FY2025?",
        },
        {
            "type": "RELEVANT",
            "question": "How many comparison operators are there in python?",
        },
        {
            "type": "IRRELEVANT",
            "question": "How many rounds a person need to run, if his BMI is 31?",
        },
        {
            "type": "IRRELEVANT",
            "question": "What is the capital of France?",
        },
    ]

    for index, test in enumerate(test_questions, start=1):

        print()
        print("=" * 70)
        print(f"TEST {index}: {test['type']}")
        print("=" * 70)

        print("Question:")
        print(test["question"])

        print()
        print("-" * 70)
        print("ANSWER")
        print("-" * 70)

        try:
            answer = rag_service.answer(
                question=test["question"]
            )

            print(answer)

        except Exception as error:
            print()
            print("TEST FAILED")
            print(type(error).__name__)
            print(str(error))

    print()
    print("=" * 70)
    print("END-TO-END RAG SERVICE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()