from app.services.embedding import EmbeddingService
from app.services.vector_store import insert_document, similarity_search


embedding_service = EmbeddingService()


documents = [
    {
        "content": (
            "GovinGPT is a government-focused generative AI system "
            "designed to provide answers using official documents."
        ),
        "metadata": {
            "source": "govingpt_test",
            "topic": "RAG",
        },
    },
    {
        "content": (
            "Delhi government services include applying for certificates, "
            "licenses, registrations, and other citizen services."
        ),
        "metadata": {
            "source": "delhi_services_test",
            "topic": "government services",
        },
    },
    {
        "content": (
            "Employees are entitled to annual leave according to the "
            "organization's leave policy. Leave requests must be submitted "
            "through the approved process."
        ),
        "metadata": {
            "source": "employee_policy_test",
            "topic": "leave policy",
        },
    },
    {
        "content": (
            "Vehicle maintenance includes regular engine oil changes, "
            "brake inspection, tyre inspection, battery checks, and "
            "scheduled servicing."
        ),
        "metadata": {
            "source": "vehicle_maintenance_test",
            "topic": "vehicle service",
        },
    },
    {
        "content": (
            "Students must follow the examination schedule published by "
            "the education authority. Examination dates and instructions "
            "are communicated before the examination period."
        ),
        "metadata": {
            "source": "education_policy_test",
            "topic": "education",
        },
    },
]


print("Inserting test documents...\n")

for document in documents:

    embedding = embedding_service.embed(
        document["content"]
    )

    document_id = insert_document(
        content=document["content"],
        embedding=embedding,
        metadata=document["metadata"],
    )

    print(
        f"Inserted document ID: {document_id} "
        f"| Topic: {document['metadata']['topic']}"
    )


queries = [
    "What is the process for vehicle maintenance?",
    "What does the employee leave policy say?",
    "What government services are available to citizens?",
    "What are the examination instructions for students?",
    "What is GovinGPT used for?",
]


for query in queries:

    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    query_embedding = embedding_service.embed(query)

    results = similarity_search(
        query_embedding=query_embedding,
        limit=3,
    )

    for rank, result in enumerate(results, start=1):

        print(
            f"\nRank {rank}"
            f"\nID: {result['id']}"
            f"\nDistance: {result['distance']}"
            f"\nTopic: {result['metadata'].get('topic', 'N/A')}"
            f"\nContent: {result['content']}"
        )