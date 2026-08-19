"""
Instrumented drop-in replacement for app/services/rag_service.py

Purpose: pinpoint EXACTLY where the ~100s per query is going, before
changing anything else. Swap this in, run a few real queries through
the Streamlit app, and check the terminal (where `streamlit run` is
running) for a per-stage timing breakdown like:

    [TIMING] embed_query      : 0.08s
    [TIMING] vector_search    : 0.02s
    [TIMING] rerank           : 4.31s
    [TIMING] context_build    : 0.00s
    [TIMING] prompt_build     : 0.00s
    [TIMING] llm_generate     : 94.70s   <-- e.g. this is your answer
    [TIMING] TOTAL            : 99.11s

Once you know which stage dominates, the fix is targeted instead of
guesswork. This file changes NO behavior, only adds timing + logging.
"""

import logging
import time
from contextlib import contextmanager

from app.services.retrieval_service import RetrievalService
from app.services.context_builder import ContextBuilder
from app.services.rag_prompt import RAGPromptBuilder
from app.services.llm import LLMService

logger = logging.getLogger("rag_timing")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[TIMING] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@contextmanager
def _timed(label: str, bucket: dict):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        bucket[label] = elapsed
        logger.info(f"{label:<20}: {elapsed:.2f}s")


class RAGService:
    """
    End-to-end RAG orchestration service.

    Flow:
        Question
            |
        Retrieval (embed + vector search + rerank)
            |
        Context Filtering
            |
        RAG Prompt
            |
        Llama
            |
        Answer
    """

    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.context_builder = ContextBuilder()
        self.prompt_builder = RAGPromptBuilder()
        self.llm_service = LLMService()

    def answer(self, question: str, top_k: int = 5) -> str:
        answer, _ = self.answer_with_sources(question=question, top_k=top_k)
        return answer

    def answer_with_sources(
        self,
        question: str,
        top_k: int = 5,
    ) -> tuple[str, list[dict]]:

        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        if not question or not question.strip():
            return "Please provide a question.", []

        question = question.strip()

        # STEP 2: Retrieve relevant document chunks (embed + search + rerank
        # are timed individually *inside* retrieval_service if you also swap
        # in retrieval_service_instrumented.py — otherwise this whole step
        # is timed as one block here).
        with _timed("retrieval_total", timings):
            retrieved_documents = self.retrieval_service.retrieve(
                query=question,
                top_k=top_k,
            )

        # STEP 3: Build filtered context
        with _timed("context_build", timings):
            context = self.context_builder.build(
                retrieved_documents=retrieved_documents
            )

        if not context:
            logger.info(f"TOTAL (no context) : {time.perf_counter() - total_start:.2f}s")
            return (
                "I could not find the answer in the provided documents.",
                retrieved_documents,
            )

        # STEP 5: Build RAG prompt
        with _timed("prompt_build", timings):
            prompt = self.prompt_builder.build_prompt(
                question=question,
                context=context,
            )
        logger.info(f"prompt_char_len     : {len(prompt)}")

        # STEP 6: Send prompt to Llama
        with _timed("llm_generate", timings):
            answer = self.llm_service.generate(prompt)

        total = time.perf_counter() - total_start
        logger.info(f"{'TOTAL':<20}: {total:.2f}s")

        return answer, retrieved_documents
