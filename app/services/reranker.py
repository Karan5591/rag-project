"""
app/services/reranker.py -- swapped to a lightweight cross-encoder

Same public interface as before: rerank(query, documents, top_k) -> list[dict]
with "reranker_score" added, sorted descending, truncated to top_k.

Why this model:
- cross-encoder/ms-marco-MiniLM-L-6-v2 (~22M params) vs.
  BAAI/bge-reranker-v2-m3 (~568M params) -> ~25x smaller.
- Trained specifically for passage reranking (MS MARCO), same task you're
  doing here, just single-language (English) instead of multilingual --
  which is fine since your corpus is English HR/financial/technical docs.
- Ships via sentence-transformers, which you already depend on for the
  embedding model -- no new dependency.

Score semantics note:
- The old FlagReranker call used normalize=True, which applies a sigmoid
  so scores land in [0, 1]. CrossEncoder's raw output is an unbounded
  logit, so this also applies a sigmoid (activation_fct=torch.nn.Sigmoid())
  to keep scores in the same [0, 1] range your existing thresholds expect:
      settings.reranker_relevance_threshold (currently 0.05)
      the hybrid_score < 0.18 cutoff in retrieval_service.py
  That said, a different model will NOT produce identical scores to the old
  one for the same query/doc pair -- only the same *range*. Re-run your
  evaluation set (test_rag_evaluation.py / test_rag_hard_cases.py) after
  swapping and re-tune those two thresholds if borderline cases start
  passing/failing differently than before.
"""

import torch
from sentence_transformers import CrossEncoder


class RerankerService:
    """
    Reranks retrieved documents using a MiniLM cross-encoder.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MAX_LENGTH = 512

    def __init__(self):
        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"

        print(f"Loading reranker model: {self.MODEL_NAME}")
        print(f"Device: {device} | max_length: {self.MAX_LENGTH}")

        self.model = CrossEncoder(
            self.MODEL_NAME,
            max_length=self.MAX_LENGTH,
            device=device,
        )

        print("Reranker model loaded successfully")

    def rerank(self, query: str, documents: list, top_k: int = 5):
        if not documents:
            return []

        pairs = [
            [query, document["content"]]
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=32,
            activation_fct=torch.nn.Sigmoid(),  # keep scores in [0, 1]
        )

        ranked = []

        for document, score in zip(documents, scores):
            item = dict(document)
            item["reranker_score"] = float(score)
            ranked.append(item)

        ranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return ranked[:top_k]
