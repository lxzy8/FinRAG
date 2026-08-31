"""
Hybrid retrieval: dense (FAISS) + BM25, fused with Reciprocal Rank Fusion.

Wrapped in a class (rather than notebook-style globals) so the embedding
model, FAISS index, and BM25 index are explicit, testable state instead of
implicit module-level variables.
"""

import numpy as np


def reciprocal_rank_fusion(rankings_list, k: int = 60) -> list:
    # RRF: score(doc) = sum over each ranking of 1/(k+rank)
    fused_scores = {}
    for ranking in rankings_list:
        for rank, doc_idx in enumerate(ranking):
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + 1 / (k + rank + 1)
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_idx for doc_idx, score in sorted_docs]


def detect_recency_intent(query: str) -> bool:
    """
    Query mein 'recent/latest/current' jaisa recency-signal hai ya nahi.
    Agar hai, to retrieval ko sirf latest filings tak restrict karna chahiye —
    warna embeddings/BM25 purane aur naye period ke chunks ko barabar relevant
    maan lete hain (recency semantic similarity se nahi, metadata se pata chalta hai).
    """
    recency_keywords = ["recent", "recently", "latest", "current", "most recent",
                         "this quarter", "last quarter", "now", "trend"]
    q = query.lower()
    return any(kw in q for kw in recency_keywords)


def detect_doc_type_intent(query: str):
    """
    Query "quarterly" maang rahi hai ya "annual" — agar clear signal hai to
    sirf usi doc_type tak target restrict karo, warna dono include karo (None).
    Isse "most recent QUARTERLY revenue" jaisi query ke liye latest 10-K
    candidate pool mein aake precision drag down nahi karega.
    """
    q = query.lower()
    if "quarterly" in q or "quarter" in q:
        return "10-Q"
    if "annual" in q or "yearly" in q or "fiscal year" in q:
        return "10-K"
    return None


def get_recent_target_docs(parsed_by_company: dict, company: str, doc_type_filter=None) -> dict:
    """
    Company ki filings mein se har doc_type (10-K, 10-Q) ka sabse recent doc_name
    nikaal ke dict return karta hai, e.g. {"10-K": "10-K_2026-01-25", "10-Q": "10-Q_2026-07-26"}.
    doc_type_filter diya ho to sirf usi type ka latest doc return hota hai
    (query ne "quarterly"/"annual" explicitly maanga ho tab).
    Period format YYYY-MM-DD hai isliye string-sort chronological hi kaam karta hai.
    """
    filings = parsed_by_company.get(company, {})
    doc_types = [doc_type_filter] if doc_type_filter else ["10-K", "10-Q"]
    result = {}
    for doc_type in doc_types:
        docs = sorted([d for d in filings if d.startswith(doc_type)])
        if docs:
            result[doc_type] = docs[-1]
    return result


class HybridRetriever:
    """
    Owns the embedding model, FAISS index, and BM25 index built over `all_chunks`.
    `search()` ranks globally; `search_within()` pre-filters to a candidate subset
    before ranking (company/period metadata filter first, ranking second) — this
    matters because post-filtering (rank globally, then discard) wastes top-k
    slots on chunks that were never eligible in the first place.
    """

    def __init__(self, all_chunks: list, embed_model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi
        import faiss

        self.all_chunks = all_chunks
        self.embed_model = SentenceTransformer(embed_model_name)

        texts_to_embed = [c["content"] for c in all_chunks]  # context-prefixed text
        self.embeddings = self.embed_model.encode(
            texts_to_embed, show_progress_bar=True, normalize_embeddings=True
        )

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity (normalized embeddings pe inner product)
        self.index.add(self.embeddings.astype("float32"))
        print(f"Index vectors: {self.index.ntotal}")

        # raw_content use karo (bina context prefix) — warna repeated "Company: X" score skew karega
        tokenized_corpus = [c["raw_content"].lower().split() for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_search(self, query: str, top_k: int = 10) -> list:
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        _, indices = self.index.search(query_embedding, top_k)
        return list(indices[0])

    def bm25_search(self, query: str, top_k: int = 10) -> list:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return list(top_idx)

    def search(self, query: str, top_k: int = 5) -> list:
        dense_results = self.dense_search(query, top_k=10)
        bm25_results = self.bm25_search(query, top_k=10)
        fused = reciprocal_rank_fusion([dense_results, bm25_results])
        return fused[:top_k]

    def search_within(self, query: str, candidate_indices, top_k: int = 10) -> list:
        """Same as search(), but ranking is restricted to `candidate_indices`."""
        if not candidate_indices:
            return []
        candidate_indices = list(candidate_indices)

        query_embedding = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")[0]
        candidate_embeddings = self.embeddings[candidate_indices]
        dense_scores = candidate_embeddings @ query_embedding
        dense_ranked = [candidate_indices[i] for i in np.argsort(dense_scores)[::-1]]

        tokenized_query = query.lower().split()
        bm25_scores_all = self.bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(candidate_indices, key=lambda idx: bm25_scores_all[idx], reverse=True)

        fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked])
        return fused[:top_k]
