"""
SEC EDGAR se filings list aur raw documents fetch karna.
"""

import requests
import time

HEADERS = {
    "User-Agent": "maxvasp18@gmail.com"  # apna email daal dena
}

BASE_URL = "https://data.sec.gov"
SUBMISSIONS_URL = f"{BASE_URL}/submissions/CIK{{cik}}.json"


def get_cik_from_ticker(ticker: str) -> str:
    """Ticker symbol (e.g. 'AAPL') se CIK number nikaalta hai."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    ticker = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker:
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker {ticker} not found")


def get_company_filings(cik: str) -> dict:
    """Company ki saari filings ki list laata hai (10-K, 10-Q, 8-K, etc.)"""
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def filter_filings_by_form(filings_data: dict, form_types: list[str]) -> list[dict]:
    """Saari filings mein se sirf specific form types filter karta hai."""
    recent = filings_data["filings"]["recent"]
    results = []

    for i in range(len(recent["form"])):
        if recent["form"][i] in form_types:
            results.append({
                "form": recent["form"][i],
                "filingDate": recent["filingDate"][i],
                "accessionNumber": recent["accessionNumber"][i],
                "primaryDocument": recent["primaryDocument"][i],
            })

    return results


def get_filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Ek specific filing ka actual document URL banata hai."""
    cik_no_padding = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_padding}/{accession_no_dashes}/{primary_document}"


def fetch_filing_document(url: str) -> str:
    """Filing ka raw HTML content fetch karta hai."""
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    time.sleep(0.15)  # SEC rate limit safe margin
    return resp.text