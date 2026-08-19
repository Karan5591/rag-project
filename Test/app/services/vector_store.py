from psycopg.types.json import Jsonb

from app.services.database import get_connection
from app.services.embedding import settings



def drop_documents_table() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS documents")
        connection.commit()

    print("Documents table dropped successfully")


def create_documents_table():
    embedding_dim = settings.embedding_dim

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        metadata JSONB,
        embedding VECTOR({embedding_dim})
    );
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(create_table_sql)

        connection.commit()

    print(
        f"Documents table created successfully "
        f"with {embedding_dim}-dimensional embeddings"
    )
def insert_document(
    content: str,
    embedding: list[float],
    metadata: dict | None = None,
) -> int:

    insert_sql = """
    INSERT INTO documents (content, metadata, embedding)
    VALUES (%s, %s, %s)
    RETURNING id;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
    insert_sql,
    (
        content,
        Jsonb(metadata) if metadata is not None else None,
        embedding,
    ),
)
            document_id = cursor.fetchone()[0]

        connection.commit()

    return document_id

def similarity_search(
    query_embedding: list[float],
    limit: int = 5,
) -> list[dict]:

    search_sql = """
    SELECT
        id,
        content,
        metadata,
        embedding <=> %s::vector AS distance
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                search_sql,
                (
                    query_embedding,
                    query_embedding,
                    limit,
                ),
            )

            rows = cursor.fetchall()

    results = []

    for row in rows:
        results.append(
            {
                "id": row[0],
                "content": row[1],
                "metadata": row[2],
                "distance": row[3],
            }
        )

    return results