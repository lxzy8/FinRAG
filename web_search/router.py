"""
Decide karta hai ki query ko web search chahiye ya sirf
corpus (SEC filings + XBRL) se answerable hai.

Rule-based rakha hai (LLM classification se fast, aur latency kam).
"""

import re

# Keywords jo indicate karte hain ki live/recent info chahiye
LIVE_INFO_KEYWORDS = [
    r"\btoday\b", r"\bcurrent(ly)?\b", r"\blatest\b", r"\brecent(ly)?\b",
    r"\bthis week\b", r"\bthis month\b", r"\bnow\b", r"\bstock price\b",
    r"\bshare price\b", r"\bmarket cap\b", r"\bnews\b", r"\banalyst\b",
    r"\bprice target\b", r"\brating\b", r"\btrading at\b",
]

# Keywords jo corpus-answerable hone ka strong signal hain
CORPUS_KEYWORDS = [
    r"\b10-k\b", r"\b10-q\b", r"\bannual report\b", r"\bquarterly report\b",
    r"\brisk factors\b", r"\bmd&a\b", r"\bfiscal year \d{4}\b", r"\bfy\d{2,4}\b",
]


def needs_web_search(query: str) -> bool:
    """
    True return karta hai agar query ko live/recent info chahiye.
    False agar corpus (filings) se hi answerable lagti hai.
    """
    query_lower = query.lower()

    # Agar explicitly filing/corpus-specific keyword hai, web search skip karo
    for pattern in CORPUS_KEYWORDS:
        if re.search(pattern, query_lower):
            return False

    # Agar live-info keyword mila, web search trigger karo
    for pattern in LIVE_INFO_KEYWORDS:
        if re.search(pattern, query_lower):
            return True

    return False  # default: corpus se hi try karo


def classify_query(query: str) -> str:
    """Debugging/logging ke liye human-readable label deta hai."""
    return "web_search" if needs_web_search(query) else "corpus_rag"