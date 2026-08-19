from app.services.rag_prompt import RAGPromptBuilder


prompt_builder = RAGPromptBuilder()


print("=" * 70)
print("RAG PROMPT BUILDER TEST")
print("=" * 70)


# ----------------------------------------------------------------------
# TEST 1: RELEVANT QUESTION
# ----------------------------------------------------------------------

question_1 = "How many unused PTO days can an employee roll over?"

context_1 = """
[Context 1]
Section: 1.1 Paid Time Off (PTO) Allocation and Roll-Over Rules

Employees may roll over a maximum of 5 unused PTO days into the
following calendar year. Any additional accrued, unused PTO beyond
5 days will be forfeited unless a written exception is granted.
"""


print("\n")
print("=" * 70)
print("TEST 1: RELEVANT QUESTION")
print("=" * 70)

prompt_1 = prompt_builder.build_prompt(
    question=question_1,
    context=context_1,
)

print(prompt_1)


# ----------------------------------------------------------------------
# TEST 2: NO RELEVANT CONTEXT
# ----------------------------------------------------------------------

question_2 = "Who is the Prime Minister of India?"

context_2 = None


print("\n")
print("=" * 70)
print("TEST 2: NO RELEVANT CONTEXT")
print("=" * 70)

prompt_2 = prompt_builder.build_prompt(
    question=question_2,
    context=context_2,
)

print(prompt_2)


print("\n")
print("=" * 70)
print("RAG PROMPT BUILDER TEST COMPLETE")
print("=" * 70)