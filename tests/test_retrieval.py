
"""
Manual smoke test — poora retrieval pipeline: chunking → embed →
hybrid search → rerank, end to end.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.edgar_fetch import (
    get_cik_from_ticker,
    get_company_filings,
    filter_filings_by_form,
    get_filing_document_url,
    fetch_filing_document,
)
from ingestion.document_parser import parse_filing
from chunking.chunker import create_chunks_with_metadata
from retrieval.hybrid_retriever import HybridIndex
from retrieval.reranker import rerank


def test_retrieval_pipeline():
    ticker = "AAPL"

    # Step 1-3: fetch, parse, chunk (pichle folders se)
    cik = get_cik_from_ticker(ticker)
    filings_data = get_company_filings(cik)
    tenk_filings = filter_filings_by_form(filings_data, ["10-K"])
    latest = tenk_filings[0]

    doc_url = get_filing_document_url(cik, latest["accessionNumber"], latest["primaryDocument"])
    html = fetch_filing_document(doc_url)
    parsed = parse_filing(html)

    chunks = create_chunks_with_metadata(
        sections=parsed["sections"],
        company_ticker=ticker,
        fiscal_year="2024",
        form_type=latest["form"],
        filing_date=latest["filingDate"],
    )
    print(f"✓ Created {len(chunks)} chunks")

    # Step 4: build hybrid index
    index = HybridIndex()
    index.build(chunks)

    # Step 5: search
    query = "What are the main risk factors related to supply chain?"
    results = index.hybrid_search(query, top_k=20)
    print(f"\n✓ Hybrid search returned {len(results)} candidates")
    print(f"Top result section: {results[0]['metadata']['section']}, score: {results[0]['hybrid_score']:.3f}")

    # Step 6: rerank
    final_results = rerank(query, results, top_k=5)
    print(f"\n✓ Reranked to top {len(final_results)}")
    for i, r in enumerate(final_results):
        print(f"\n--- Rank {i+1} (score: {r['rerank_score']:.3f}, section: {r['metadata']['section']}) ---")
        print(r["text"][:200])


if __name__ == "__main__":
    test_retrieval_pipeline()