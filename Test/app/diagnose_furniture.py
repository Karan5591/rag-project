"""
Diagnostic: for each failing question, show what similarity_search
actually returns (stage 1, unfiltered) and what the reranker scores
each candidate (stage 2), bypassing RetrievalService's all-or-nothing
threshold gate so we can see exactly where content is being rejected.

Run with:
    python diagnose_retrieval_gap.py
"""

import os
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHINDUCTOR_SKIP_AUTOGRAD_FALLBACK"] = "1"
import torch
torch.compile = lambda x: x

from app.services.embedding import EmbeddingService
from app.services.vector_store import similarity_search
from app.services.reranker import RerankerService
from app.services.llm import settings

QUESTIONS = [
    "what are working hour?",
    "What is employee salary provided?",
    "is there any conditions that can cause Termination?",
]

embedding_service = EmbeddingService()
reranker_service = RerankerService()

print(f"Current reranker_relevance_threshold: {settings.reranker_relevance_threshold}")
print(f"retrieval_top_k: {settings.retrieval_top_k}  final_top_k: {settings.final_top_k}")

for question in QUESTIONS:
    print()
    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    query_embedding = embedding_service.embed(question)

    # Stage 1: raw vector similarity, unfiltered
    results = similarity_search(query_embedding=query_embedding, limit=settings.retrieval_top_k)

    print(f"\nSTAGE 1 — raw similarity_search ({len(results)} candidates):")
    if not results:
        print("  ⚠️ NOTHING returned at all — problem is embedding/vector-store level, not reranking.")
        continue

    for r in results:
        section = (r.get("metadata") or {}).get("section")
        content_type = (r.get("metadata") or {}).get("content_type")
        preview = (r.get("content") or "")[:70].replace("\n", " ")
        print(f"  dist={r.get('distance'):.4f}  section={section!r}  type={content_type!r}  {preview!r}")

    # Stage 2: reranker, unfiltered (no threshold applied)
    reranked = reranker_service.rerank(query=question, documents=results, top_k=len(results))

    print(f"\nSTAGE 2 — reranked, ALL candidates shown (threshold NOT applied here):")
    for r in reranked:
        score = r.get("reranker_score")
        section = (r.get("metadata") or {}).get("section")
        preview = (r.get("content") or "")[:70].replace("\n", " ")
        flag = "PASSES threshold" if score >= settings.reranker_relevance_threshold else "would be REJECTED"
        print(f"  score={score:.4f}  [{flag}]  section={section!r}  {preview!r}")

print()
print("=" * 70)
print("READING THIS:")
print("- If Stage 1 never shows the right chunk in its list at all,")
print("  the problem is upstream: embeddings/vector similarity, not reranking.")
print("- If Stage 1 DOES show the right chunk, but Stage 2's score for it")
print("  is below the threshold, the problem is the reranker threshold")
print("  being too strict for this chunk/question pairing.")
print("=" * 70)