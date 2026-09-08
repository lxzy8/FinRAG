"""
Web search API wrapper — DuckDuckGo (free, no key) aur Tavily
(better quality, needs API key) dono support karta hai.
"""

import os
from duckduckgo_search import DDGS


def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """
    Free, no API key chahiye. Development/testing ke liye theek hai.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return results


def search_tavily(query: str, max_results: int = 5) -> list[dict]:
    """
    Better quality, LLM-agent-specific results. TAVILY_API_KEY env var chahiye.
    """
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=max_results)

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        })
    return results


def web_search(query: str, max_results: int = 5, provider: str = "duckduckgo") -> list[dict]:
    """Unified entry point — provider switch karne ke liye."""
    if provider == "tavily":
        return search_tavily(query, max_results)
    return search_duckduckgo(query, max_results)