
"""
Router + search client test — Mac pe ye pura chal sakta hai
(MiniCPM5-1B chhota hai, CPU pe bhi chal jayega, thoda slow hoga).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_search.router import needs_web_search, classify_query
from web_search.search_client import web_search
from web_search.content_filter import filter_and_summarize


def test_router():
    test_queries = [
        "What is Apple's current stock price?",
        "What were the risk factors mentioned in Apple's 10-K?",
        "Latest news on Tesla's earnings",
        "What was the fiscal year 2024 revenue for Microsoft?",
    ]

    print("=== Router classification ===")
    for q in test_queries:
        print(f"  '{q}' → {classify_query(q)}")


def test_search_and_filter():
    query = "Apple stock price today"

    print(f"\n=== Searching: '{query}' ===")
    results = web_search(query, max_results=5, provider="duckduckgo")
    print(f"✓ Found {len(results)} results")
    for r in results[:2]:
        print(f"  - {r['title']}: {r['url']}")

    print("\n=== Filtering with MiniCPM5-1B ===")
    filtered = filter_and_summarize(query, results)
    print(filtered)


if __name__ == "__main__":
    test_router()
    test_search_and_filter()