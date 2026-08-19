from sentence_transformers import SentenceTransformer
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_model: str
    embedding_dim: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


class EmbeddingService:
    def __init__(self):
        self.model_name = settings.embedding_model
        self.embedding_dim = settings.embedding_dim

        print(f"Loading embedding model: {self.model_name}")

        self.model = SentenceTransformer(self.model_name)

        print(
            f"Embedding model loaded successfully "
            f"({self.embedding_dim} dimensions)"
        )

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)

        return embedding.tolist()