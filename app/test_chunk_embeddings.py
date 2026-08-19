from pathlib import Path

from docling.document_converter import DocumentConverter

from app.services.chunking_service import ChunkingService
from app.services.embedding import EmbeddingService


PDF_PATH = Path("data/raw/test_document.pdf")


print("=" * 70)
print("CHUNK → EMBEDDING PIPELINE TEST")
print("=" * 70)


# ---------------------------------------------------------
# STEP 1: Document conversion
# ---------------------------------------------------------

print("\nSTEP 1: DOCUMENT CONVERSION")
print("-" * 70)

if not PDF_PATH.exists():
    raise FileNotFoundError(f"File not found: {PDF_PATH}")

print(f"File: {PDF_PATH}")
print("Converting document...")

converter = DocumentConverter()
result = converter.convert(PDF_PATH)

document = result.document

print("Document processed successfully.")


# ---------------------------------------------------------
# STEP 2: Structure-aware chunking
# ---------------------------------------------------------

print("\nSTEP 2: STRUCTURE-AWARE CHUNKING")
print("-" * 70)

chunking_service = ChunkingService(
    chunk_size=800,
    chunk_overlap=100,
)

chunks = chunking_service.chunk_docling_document(
    document,
    document_id="DOC-TEST-001",
    document_name="test_document",
)

print(f"Total chunks: {len(chunks)}")


# ---------------------------------------------------------
# STEP 3: Load embedding service
# ---------------------------------------------------------

print("\nSTEP 3: EMBEDDING MODEL")
print("-" * 70)

embedding_service = EmbeddingService()


# ---------------------------------------------------------
# STEP 4: Generate embeddings
# ---------------------------------------------------------

print("\nSTEP 4: GENERATING EMBEDDINGS")
print("-" * 70)

embedded_chunks = []
failed_chunks = []

for chunk in chunks:

    try:
        embedding = embedding_service.embed(chunk.content)

        embedded_chunks.append(
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": embedding,
            }
        )

        print(
            f"Chunk {chunk.chunk_id:02d} "
            f"→ {len(embedding)} dimensions"
        )

    except Exception as e:

        failed_chunks.append(
            {
                "chunk_id": chunk.chunk_id,
                "error": str(e),
            }
        )

        print(
            f"Chunk {chunk.chunk_id:02d} "
            f"→ FAILED: {e}"
        )


# ---------------------------------------------------------
# STEP 5: Validation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("EMBEDDING VALIDATION")
print("=" * 70)

print(f"Total chunks       : {len(chunks)}")
print(f"Successful         : {len(embedded_chunks)}")
print(f"Failed             : {len(failed_chunks)}")

dimensions = set(
    len(item["embedding"])
    for item in embedded_chunks
)

print(f"Embedding dimensions found: {dimensions}")


# ---------------------------------------------------------
# Validation checks
# ---------------------------------------------------------

if len(embedded_chunks) != len(chunks):
    print("\n❌ NOT ALL CHUNKS WERE EMBEDDED")

else:
    print("\n✅ All chunks embedded successfully")


if dimensions == {embedding_service.embedding_dim}:
    print(
        f"✅ All embeddings have "
        f"{embedding_service.embedding_dim} dimensions"
    )

else:
    print(
        "❌ Embedding dimension mismatch"
    )


if all(
    isinstance(value, float)
    for item in embedded_chunks
    for value in item["embedding"]
):
    print("✅ All embedding values are numeric")

else:
    print("❌ Invalid embedding values detected")


if all(
    item["content"].strip()
    for item in embedded_chunks
):
    print("✅ No empty chunk content")

else:
    print("❌ Empty chunk content detected")


print("\n" + "=" * 70)
print("CHUNK → EMBEDDING PIPELINE TEST COMPLETE")
print("=" * 70)