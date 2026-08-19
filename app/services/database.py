import psycopg
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
EVALUATION_DATASET = [
    {
        "id": "T001",
        "category": "direct_relevant",
        "question": "How many unused PTO days can an employee roll over?",
        "expected": "answer",
        "expected_answer": "5",
    },
    {
        "id": "T002",
        "category": "direct_relevant",
        "question": "What does a flashing amber LED indicate?",
        "expected": "answer",
        "expected_answer": "fan failure or thermal threshold exceeded",
    },
    {
        "id": "T003",
        "category": "direct_relevant",
        "question": "What is the maximum throughput of the Edge Router?",
        "expected": "answer",
        "expected_answer": "18.4 Gbps",
    },

    # ...
]

def get_connection():
    return psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )