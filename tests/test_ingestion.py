"""
Manual smoke test — ingestion folder ke saare pieces
end-to-end chala ke dekhte hain ki sab connect ho raha hai.
"""

import sys
import os

# Fin/ (project root) ko path mein add karo, taaki 'ingestion' package mil jaye
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.edgar_fetch import (
    get_cik_from_ticker,
    get_company_filings,
    filter_filings_by_form,
    get_filing_document_url,
    fetch_filing_document,
)
from ingestion.xbrl_parser import get_xbrl_facts, extract_metric, save_json
from ingestion.document_parser import parse_filing


def test_edgar_fetch():
    ticker = "AAPL"
    cik = get_cik_from_ticker(ticker)
    print(f"✓ {ticker} CIK: {cik}")

    filings_data = get_company_filings(cik)
    tenk_filings = filter_filings_by_form(filings_data, ["10-K"])
    print(f"✓ Found {len(tenk_filings)} 10-K filings")

    return cik, tenk_filings


def test_xbrl_parser(cik: str):
    xbrl_data = get_xbrl_facts(cik)
    revenue = extract_metric(xbrl_data, "Revenues")
    print(f"✓ Revenue entries found: {len(revenue)}")
    if revenue:
        print(f"  Sample: {revenue[0]}")

    save_json(xbrl_data, "data/raw/AAPL_xbrl_facts.json")
    print("✓ Saved XBRL data to data/raw/")


def test_document_parser(cik: str, tenk_filings: list):
    latest = tenk_filings[0]
    doc_url = get_filing_document_url(
        cik, latest["accessionNumber"], latest["primaryDocument"]
    )
    print(f"✓ Fetching document: {doc_url}")

    html = fetch_filing_document(doc_url)
    parsed = parse_filing(html)

    print(f"✓ Parsed text length: {parsed['full_text_length']}")
    print(f"✓ Sections found: {parsed['sections_found']}")

    if "mda" in parsed["sections"]:
        print("\n--- MD&A preview ---")
        print(parsed["sections"]["mda"][:300])


if __name__ == "__main__":
    print("=== Testing edgar_fetch.py ===")
    cik, tenk_filings = test_edgar_fetch()

    print("\n=== Testing xbrl_parser.py ===")
    test_xbrl_parser(cik)

    print("\n=== Testing document_parser.py ===")
    test_document_parser(cik, tenk_filings)