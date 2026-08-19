import re

from app.services.embedding import EmbeddingService
from app.services.vector_store import similarity_search
from app.services.reranker import RerankerService
from app.services.llm import settings


class RetrievalService:
    """
    Service responsible for retrieving relevant document
    chunks for a user query.

    Retrieval happens in two stages:

    1. Vector similarity search retrieves a broad candidate set.
    2. BGE-Reranker reorders the candidates by query relevance.
    """

    def __init__(
        self,
        retrieval_top_k: int = 10,
        final_top_k: int = 5,
    ):
        self.embedding_service = EmbeddingService()
        self.reranker_service = RerankerService()

        self.retrieval_top_k = settings.retrieval_top_k
        self.final_top_k = settings.final_top_k

    @staticmethod
    def _normalize_text(value: str) -> str:
        value = (value or "").lower()
        value = value.replace("renumeration", "remuneration")
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @classmethod
    def _expand_query_terms(cls, query: str) -> set[str]:
        text = cls._normalize_text(query)
        if not text:
            return set()

        tokens = set(text.split())
        expansions = set(tokens)
        synonym_map = {
            "remuneration": {"remuneration", "salary", "pay", "monthly"},
            "salary": {"salary", "remuneration", "pay"},
            "leave": {"leave", "leaves", "vacation", "holiday"},
            "hours": {"hours", "hour", "timing", "working"},
            "contact": {"contact", "email", "phone", "website", "cin"},
            "joining": {"joining", "join", "joined", "appointment"},
        }

        for token in list(tokens):
            for key, values in synonym_map.items():
                if token == key or token in values:
                    expansions.update(values)

        return expansions

    @classmethod
    def _keyword_overlap_score(cls, query: str, content: str) -> float:
        query_terms = cls._expand_query_terms(query)
        content_terms = cls._expand_query_terms(content)

        if not query_terms or not content_terms:
            return 0.0

        overlap = query_terms & content_terms
        return len(overlap) / max(len(query_terms), 1)

    @classmethod
    def _field_match_bonus(cls, query: str, content: str, metadata: dict | None) -> float:
        field_name = (metadata or {}).get("field_name")
        query_text = cls._normalize_text(query)
        content_text = cls._normalize_text(content)

        if field_name and field_name in query_text:
            return 0.25
        if "remuneration" in query_text and "remuneration" in content_text:
            return 0.2
        if "joining" in query_text and "joining" in content_text:
            return 0.2
        if "leave" in query_text and "leave" in content_text:
            return 0.15
        if "contact" in query_text and ("email" in content_text or "website" in content_text or "cin" in content_text):
            return 0.18
        return 0.0

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """Retrieve and rerank relevant document chunks."""

        if not query or not query.strip():
            return []

        query = query.strip()

        query_embedding = self.embedding_service.embed(query)
        retrieval_limit = max(self.retrieval_top_k, 20)

        results = similarity_search(
            query_embedding=query_embedding,
            limit=retrieval_limit,
        )

        if not results:
            return []

        for item in results:
            content = str(item.get("content", ""))
            item["keyword_overlap"] = self._keyword_overlap_score(query, content)
            item["field_bonus"] = self._field_match_bonus(
                query=query,
                content=content,
                metadata=item.get("metadata"),
            )
            item["hybrid_score"] = item["keyword_overlap"] + item["field_bonus"]

        final_limit = top_k if top_k is not None else self.final_top_k
        reranked_results = self.reranker_service.rerank(
            query=query,
            documents=results,
            top_k=final_limit,
        )

        if not reranked_results:
            return []

        for item in reranked_results:
            item["keyword_overlap"] = item.get("keyword_overlap", 0.0)
            item["field_bonus"] = item.get("field_bonus", 0.0)
            item["hybrid_score"] = (
                float(item.get("reranker_score", 0.0))
                + item.get("keyword_overlap", 0.0)
                + item.get("field_bonus", 0.0)
            )

        reranked_results.sort(
            key=lambda item: item.get("hybrid_score", 0.0),
            reverse=True,
        )

        best_score = reranked_results[0].get("reranker_score", 0.0)
        best_hybrid = reranked_results[0].get("hybrid_score", 0.0)

        if best_score < settings.reranker_relevance_threshold:
            if best_hybrid < 0.18:
                return []

        return reranked_results[:final_limit]