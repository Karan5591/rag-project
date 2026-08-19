from app.services.retrieval_service import RetrievalService


print("=" * 70)
print("RETRIEVAL SERVICE TEST")
print("=" * 70)


retrieval_service = RetrievalService()


queries = [
    {
        "type": "RELEVANT",
        "question": "How many unused PTO days can an employee roll over?",
    },
    {
        "type": "RELEVANT",
        "question": "How many rounds a person need to run, if his BMI is 31?",
    },
    
    {
        "type": "IRRELEVANT",
        "question": "How many comparison operators are there in python?",
    },
]


for index, item in enumerate(queries, start=1):

    question_type = item["type"]
    question = item["question"]

    print("\n" + "=" * 70)
    print(f"QUERY {index}")
    print("=" * 70)

    print(f"Type     : {question_type}")
    print(f"Question : {question}")

    results = retrieval_service.retrieve(
        query=question,
        top_k=5,
    )

    print(f"Retrieved: {len(results)}")

    for rank, result in enumerate(results, start=1):

        metadata = result.get("metadata") or {}

        print("\n" + "-" * 70)

        print(f"Rank       : {rank}")
        print(f"Database ID: {result['id']}")
        print(f"Distance   : {result['distance']}")
        print(f"Section    : {metadata.get('section')}")
        print(f"Document   : {metadata.get('document_name')}")
        print(f"Content Type: {metadata.get('content_type')}")
        print(f"Content:\n{result['content']}")


print("\n" + "=" * 70)
print("RETRIEVAL SERVICE TEST COMPLETE")
print("=" * 70)