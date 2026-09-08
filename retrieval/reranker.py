"""
Cross-encoder reranker (bge-reranker-v2-m3) — hybrid search
ke top candidates ko final precision ke liye rerank karta hai.
"""

from FlagEmbedding import FlagReranker

_reranker = None


def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        print("Loading bge-reranker-v2-m3...")
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _reranker


def rerank(query: str, chunks: list[dict], top_k: int = 8) -> list[dict]:
    """
    Query + chunk pairs ko cross-encoder se score karta hai,
    aur top_k sabse relevant chunks return karta hai.
    """
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in chunks]
    scores = reranker.compute_score(pairs, normalize=True)

    # Agar sirf ek chunk hai, scores ek single float ho sakta hai (list nahi)
    if isinstance(scores, float):
        scores = [scores]

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks[:top_k]