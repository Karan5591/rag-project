from typing import Any


class ContextBuilder:
    """
    Builds the final LLM context from retrieved+reranked chunks.

    NOTE: relevance is already decided upstream, in RetrievalService,
    by the BGE reranker score (see settings.reranker_relevance_threshold).
    That's a query-aware relevance signal. This class previously also
    filtered again on raw vector distance (max_distance=0.70) — a
    *second*, independent, non-query-aware gate. The two don't always
    agree: a chunk the reranker correctly scores as relevant to THIS
    question can still have a middling raw cosine distance, and would
    get silently dropped here after already passing the better check.
    That's a plausible cause of "retriever found it, answer didn't
    have it" failures (e.g. the PTO exception clause, T003/T008-style
    low-margin matches).

    max_distance now defaults to None (no second filter). Pass a float
    explicitly only if you have a specific reason to re-add a raw-
    distance safety net on top of reranking.
    """

    def __init__(self, max_distance: float | None = None):
        self.max_distance = max_distance

    def build(
        self,
        retrieved_documents: list[dict[str, Any]],
    ) -> str:

        if self.max_distance is not None:
            relevant_documents = [
                document
                for document in retrieved_documents
                if document.get("distance") is not None
                and document["distance"] <= self.max_distance
            ]
        else:
            relevant_documents = retrieved_documents

        if not relevant_documents:
            return ""

        context_parts = []

        for index, document in enumerate(relevant_documents, start=1):
            section = document.get("metadata", {}).get("section")
            content = document.get("content", "").strip()

            if not content:
                continue

            context_parts.append(
                f"[Context {index}]\n"
                f"Section: {section or 'Unknown'}\n"
                f"{content}"
            )

        return "\n\n".join(context_parts)