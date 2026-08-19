import time

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    llm_api_key: str
    llm_api_base_url: str
    llm_model: str
    llm_endpoint: str

    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    llm_timeout: int = 600
    llm_retry_count: int = 2
    reranker_relevance_threshold: float = 0.05
    retrieval_top_k: int = 10
    final_top_k: int = 5
    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = LLMSettings()


class LLMService:
    """
    Handles communication with the team's Llama gateway.
    """

    def __init__(self):
        self.url = (
            settings.llm_api_base_url.rstrip("/")
            + settings.llm_endpoint
        )

        self.headers = {
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, prompt: str) -> str:
        payload = {
            "model": settings.llm_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }

        last_error = None

        for attempt in range(settings.llm_retry_count + 1):
            try:
                with httpx.Client(
                    timeout=settings.llm_timeout
                ) as client:

                    response = client.post(
                        self.url,
                        headers=self.headers,
                        json=payload,
                    )

                    response.raise_for_status()

                    data = response.json()

                    return data["choices"][0]["message"]["content"].strip()

            except Exception as error:
                last_error = error

                if attempt < settings.llm_retry_count:
                    time.sleep(2)

        raise RuntimeError(
            f"LLM request failed: {last_error}"
        )