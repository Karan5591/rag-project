from app.services.embedding import EmbeddingService


embedding_service = EmbeddingService()


text = "This is a test document for GovinGPT RAG."

embedding = embedding_service.embed(text)


print("Embedding generated successfully")
print("Dimensions:", len(embedding))
print("First 5 values:", embedding[:5])