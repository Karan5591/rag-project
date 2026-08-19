from app.services.embedding import EmbeddingService
from app.services.vector_store import similarity_search
from app.services.context_builder import ContextBuilder


embedding_service = EmbeddingService()
context_builder = ContextBuilder(max_distance=0.70)


queries = [
    {
        "type": "RELEVANT",
        "question": "How many unused PTO days can an employee roll over?",
    },
    {
        "type": "RELEVANT",
        "question": "What was the adjusted EBITDA for FY2025?",
    },
    {
        "type": "IRRELEVANT",
        "question": "Who is the Prime Minister of India?",
    },
]


print("=" * 70)
print("CONTEXT BUILDER TEST")
print("=" * 70)


for index, item in enumerate(queries, start=1):

    question = item["question"]

    print(f"\n{'=' * 70}")
    print(f"QUERY {index}")
    print(f"{'=' * 70}")

    print("Type     :", item["type"])
    print("Question :", question)

    query_embedding = embedding_service.embed(question)

    results = similarity_search(
        query_embedding=query_embedding,
        limit=5,
    )

    print("Retrieved:", len(results))

    context = context_builder.build(results)

    print("\n" + "-" * 70)
    print("FILTERED CONTEXT")
    print("-" * 70)

    if context:
        print(context)
    else:
        print("No relevant context found.")

    print("\nContext available:", bool(context))


print("\n" + "=" * 70)
print("CONTEXT BUILDER TEST COMPLETE")
print("=" * 70)