from app.services.reranker import RerankerService


def main():
    print("=" * 70)
    print("BGE RERANKER TEST")
    print("=" * 70)

    query = "What does a flashing amber LED indicate?"

    documents = [
        {
            "content": (
                "When troubleshooting the Apex Edge Router X-9000, "
                "Flashing Amber (2 Hz): Fan failure or thermal threshold "
                "exceeded (>75°C internal core temp)."
            ),
            "section": "2.2 Diagnostic LED Light Sequence Rules",
        },
        {
            "content": (
                "The committee discussed APAC expansion acceleration, "
                "technical debt reduction for the Edge Router firmware "
                "codebase, and preparations for the annual ISO/IEC 27001 audit."
            ),
            "section": "6.1 Summary of Discussions",
        },
        {
            "content": (
                "Adjusted EBITDA closed at $98.4M, representing an EBITDA "
                "margin of 23.85%."
            ),
            "section": "4.2 Key Financial Ratios & EBITDA",
        },
        {
            "content": (
                "Employees may roll over a maximum of 5 unused PTO days "
                "into the following calendar year."
            ),
            "section": "1.1 Paid Time Off",
        },
        {
            "content": (
                "The action item matrix includes deployment of firmware "
                "hotfix 4.2.1 addressing an LED status reporting glitch."
            ),
            "section": "6.2 Action Item Matrix",
        },
    ]

    print()
    print("Query:")
    print(query)

    print()
    print("-" * 70)
    print("LOADING RERANKER")
    print("-" * 70)

    reranker = RerankerService()

    print()
    print("-" * 70)
    print("RERANKING RESULTS")
    print("-" * 70)

    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=5,
    )

    for rank, document in enumerate(results, start=1):
        print()
        print(f"Rank       : {rank}")
        print(f"Score      : {document['reranker_score']}")
        print(f"Section    : {document['section']}")
        print(f"Content    : {document['content']}")

    print()
    print("=" * 70)
    print("BGE RERANKER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()