"""
bge-m3 embedding model wrapper — dense vector retrieval ke liye.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_model = None  # lazy load, ek hi baar model load ho


import torch

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading bge-m3 embedding model on {device}...")
        _model = SentenceTransformer("BAAI/bge-m3", device=device)
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    List of texts ko embeddings mein convert karta hai.
    Chunks ko batch mein embed karne ke liye use hoga (ingestion time pe).
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity ke liye zaroori
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Single query ko embed karta hai (retrieval time pe)."""
    model = get_embedding_model()
    return model.encode([query], normalize_embeddings=True)[0]