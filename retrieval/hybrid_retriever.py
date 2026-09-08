"""
Dense (bge-m3) + Sparse (BM25) retrieval ko combine karna,
aur ek vector store (in-memory ya Qdrant) mein search karna.
"""

import numpy as np

from retrieval.embeddings import embed_texts, embed_query
from retrieval.bm25_index import build_bm25_index, bm25_search


class HybridIndex:
    """
    Ek simple in-memory hybrid index — chhote/medium corpus ke liye theek hai.
    (Bade corpus ke liye baad mein Qdrant se replace karenge.)
    """

    def __init__(self):
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.bm25 = None

    def build(self, chunks: list[dict]):
        """Saare chunks se dense embeddings aur BM25 index banata hai."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        print(f"Building embeddings for {len(chunks)} chunks...")
        self.embeddings = embed_texts(texts)

        print("Building BM25 index...")
        self.bm25 = build_bm25_index(chunks)

        print("Hybrid index ready.")

    def dense_search(self, query: str, top_k: int = 50) -> list[dict]:
        """Cosine similarity se dense search."""
        q_emb = embed_query(query)
        scores = self.embeddings @ q_emb  # normalized hain, so dot product = cosine sim

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            result = dict(self.chunks[idx])
            result["dense_score"] = float(scores[idx])
            results.append(result)

        return results

    def hybrid_search(self, query: str, top_k: int = 50, alpha: float = 0.5) -> list[dict]:
        """
        Dense aur sparse scores ko combine karta hai.
        alpha: dense ka weight (0.5 = both equal, 1.0 = pure dense, 0.0 = pure BM25)
        """
        dense_results = {c["chunk_id"]: c for c in self.dense_search(query, top_k=100)}
        sparse_results = {c["chunk_id"]: c for c in bm25_search(self.bm25, self.chunks, query, top_k=100)}

        # Scores ko normalize karo (0-1 range mein) taaki fair combine ho sake
        all_ids = set(dense_results.keys()) | set(sparse_results.keys())

        dense_scores = {cid: c.get("dense_score", 0) for cid, c in dense_results.items()}
        sparse_scores = {cid: c.get("bm25_score", 0) for cid, c in sparse_results.items()}

        max_dense = max(dense_scores.values()) if dense_scores else 1
        max_sparse = max(sparse_scores.values()) if sparse_scores else 1

        combined = []
        for cid in all_ids:
            d_score = dense_scores.get(cid, 0) / (max_dense or 1)
            s_score = sparse_scores.get(cid, 0) / (max_sparse or 1)
            final_score = alpha * d_score + (1 - alpha) * s_score

            chunk = dense_results.get(cid) or sparse_results.get(cid)
            chunk = dict(chunk)
            chunk["hybrid_score"] = final_score
            combined.append(chunk)

        combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return combined[:top_k]