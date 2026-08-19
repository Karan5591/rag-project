from FlagEmbedding import FlagReranker


class RerankerService:
    """
    Reranks retrieved documents using BGE-Reranker.
    """

    MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(self):
        print(f"Loading reranker model: {self.MODEL_NAME}")

        self.model = FlagReranker(
            self.MODEL_NAME,
            use_fp16=False,
        )

        print("Reranker model loaded successfully")

    def rerank(self, query: str, documents: list, top_k: int = 5):
        if not documents:
            return []

        pairs = [
            [query, document["content"]]
            for document in documents
        ]

        scores = self.model.compute_score(
            pairs,
            normalize=True,
        )

        if not isinstance(scores, list):
            scores = [scores]

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