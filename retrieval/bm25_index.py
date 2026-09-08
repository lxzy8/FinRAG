"""
BM25 sparse retrieval — keyword-based exact match ke liye,
dense embeddings ke saath hybrid combine hoga.
"""

import re
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer BM25 ke liye."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Chunks list se BM25 index banata hai.
    chunks: [{"chunk_id": ..., "text": ..., "metadata": ...}, ...]
    """
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def bm25_search(bm25: BM25Okapi, chunks: list[dict], query: str, top_k: int = 50) -> list[dict]:
    """Query ke liye top-k BM25 matches return karta hai, scores ke saath."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored_chunks[:top_k]:
        result = dict(chunk)
        result["bm25_score"] = float(score)
        results.append(result)

    return results


def save_bm25_index(bm25: BM25Okapi, chunks: list[dict], filepath: str):
    """BM25 index aur chunks dono ko disk pe save karta hai (Kaggle persistence ke liye)."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)


def load_bm25_index(filepath: str) -> tuple[BM25Okapi, list[dict]]:
    """Saved BM25 index load karta hai."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]