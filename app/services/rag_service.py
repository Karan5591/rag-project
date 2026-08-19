from app.services.retrieval_service import RetrievalService
from app.services.context_builder import ContextBuilder
from app.services.rag_prompt import RAGPromptBuilder
from app.services.llm import LLMService


class RAGService:
    """
    End-to-end RAG orchestration service.

    Flow:
        Question
            ↓
        Retrieval
            ↓
        Context Filtering
            ↓
        RAG Prompt
            ↓
        Llama
            ↓
        Answer
    """

    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.context_builder = ContextBuilder()
        self.prompt_builder = RAGPromptBuilder()
        self.llm_service = LLMService()

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> str:

        answer, _ = self.answer_with_sources(
            question=question,
            top_k=top_k,
        )

        return answer

    def answer_with_sources(
        self,
        question: str,
        top_k: int = 5,
    ) -> tuple[str, list[dict]]:

        # --------------------------------------------------
        # STEP 1: Validate question
        # --------------------------------------------------
        if not question or not question.strip():
            return "Please provide a question.", []

        question = question.strip()

        # --------------------------------------------------
        # STEP 2: Retrieve relevant document chunks
        # --------------------------------------------------
        retrieved_documents = self.retrieval_service.retrieve(
            query=question,
            top_k=top_k,
        )

        # --------------------------------------------------
        # STEP 3: Build filtered context
        # --------------------------------------------------
        context = self.context_builder.build(
            retrieved_documents=retrieved_documents
        )

        # --------------------------------------------------
        # STEP 4: No relevant context
        # --------------------------------------------------
        if not context:
            return (
                "I could not find the answer in the provided documents.",
                retrieved_documents,
            )

        # --------------------------------------------------
        # STEP 5: Build RAG prompt
        # --------------------------------------------------
        prompt = self.prompt_builder.build_prompt(
            question=question,
            context=context,
        )

        # --------------------------------------------------
        # STEP 6: Send prompt to Llama
        # --------------------------------------------------
        answer = self.llm_service.generate(prompt)

        return answer, retrieved_documents