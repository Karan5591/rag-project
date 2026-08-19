from app.services.llm import LLMService


print("=" * 70)
print("LLM SERVICE TEST")
print("=" * 70)


prompt = """
You are GovinGPT.

Answer only from the provided context.

Context:
Employees may roll over a maximum of 5 unused PTO days.

Question:
How many unused PTO days can an employee roll over?
"""


try:
    llm = LLMService()

    answer = llm.generate(prompt)

    print("\nLLM RESPONSE\n")
    print("-" * 70)
    print(answer)

except Exception as error:
    print("\nTEST FAILED\n")
    print(type(error).__name__)
    print(error)


print("\n" + "=" * 70)
print("LLM SERVICE TEST COMPLETE")
print("=" * 70)