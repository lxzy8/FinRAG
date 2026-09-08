"""
Manual smoke test — ingestion + chunking dono ek saath chala ke
dekhte hain ki poora flow (fetch → parse → chunk) kaam kar raha hai.
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
from chunking.chunker import create_chunks_with_metadata, print_chunk_stats


def test_full_chunking_flow():
    ticker = "AAPL"

    # Step 1: Fetch filing (ingestion se)
    cik = get_cik_from_ticker(ticker)
    filings_data = get_company_filings(cik)
    tenk_filings = filter_filings_by_form(filings_data, ["10-K"])
    latest = tenk_filings[0]

    doc_url = get_filing_document_url(
        cik, latest["accessionNumber"], latest["primaryDocument"]
    )
    html = fetch_filing_document(doc_url)
    print(f"✓ Fetched filing: {doc_url}")

    # Step 2: Parse into sections (ingestion se)
    parsed = parse_filing(html)
    print(f"✓ Parsed sections: {parsed['sections_found']}")

    # Step 3: Chunk with metadata (naya chunking module)
    chunks = create_chunks_with_metadata(
        sections=parsed["sections"],
        company_ticker=ticker,
        fiscal_year="2024",  # abhi hardcoded, baad mein filing date se derive karenge
        form_type=latest["form"],
        filing_date=latest["filingDate"],
    )
    print(f"✓ Created {len(chunks)} chunks")

    print("\n--- Chunk stats ---")
    print_chunk_stats(chunks)

    print("\n--- Sample chunk ---")
    print(chunks[0])


if __name__ == "__main__":
    test_full_chunking_flow()