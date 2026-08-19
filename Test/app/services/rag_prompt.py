class RAGPromptBuilder:
    """
    Builds the final prompt for the LLM using
    retrieved and filtered RAG context.
    """

    SYSTEM_INSTRUCTION = """
You are GovinGPT, a government-focused AI assistant.

Your task is to answer the user's question using ONLY the information
provided in the context.

Rules:
1. Use only the provided context to answer the question.
2. Do not use outside knowledge.
3. Do not make up or infer information that is not present in the context.
4. If the answer cannot be found in the context, clearly state:
   "I could not find the answer in the provided documents."
5. Keep the answer clear, concise, and directly relevant to the question.
""".strip()

    def build_prompt(
        self,
        question: str,
        context: str | None,
    ) -> str:

        if not context:
            context = "No relevant context was found in the provided documents."

        prompt = f"""
{self.SYSTEM_INSTRUCTION}

======================================================================
CONTEXT
======================================================================

{context}

======================================================================
QUESTION
======================================================================

{question}

======================================================================
ANSWER
======================================================================

""".strip()

        return prompt