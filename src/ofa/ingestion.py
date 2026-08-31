"""SEC EDGAR download + filing parsing (HTML tables + plain text extraction)."""

import os
import re

from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader


def download_filings(companies: dict, sec_email: str, k_limit: int = 2, q_limit: int = 4,
                      app_name: str = "OpenFinancialAnalyst") -> None:
    """companies: {display_name: ticker}. Downloads 10-Ks and 10-Qs for each ticker."""
    dl = Downloader(app_name, sec_email)
    for name, ticker in companies.items():
        print(f"Downloading {name} ({ticker})...")
        dl.get("10-K", ticker, limit=k_limit)
        dl.get("10-Q", ticker, limit=q_limit)


def get_filing_period(filepath: str) -> str:
    """
    Har SEC filing ke header mein 'CONFORMED PERIOD OF REPORT' hota hai.
    Isse hum manually guess nahi karte, seedha file se actual date nikalte hain.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.read(3000)
    match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", header)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return "unknown"


def build_filing_paths(ticker: str, folder: str = "sec-edgar-filings") -> dict:
    """
    Ek ticker ke saare downloaded filings dhoondh ke, unka actual period
    (header se) nikaal ke, ek clean naam wala dict banata hai: {"10-K_2025-12-28": path}.
    """
    paths = {}
    for doc_type in ["10-K", "10-Q"]:
        base = f"{folder}/{ticker}/{doc_type}"
        if not os.path.exists(base):
            continue
        for accession_folder in os.listdir(base):
            filepath = os.path.join(base, accession_folder, "full-submission.txt")
            if os.path.exists(filepath):
                period = get_filing_period(filepath)
                key = f"{doc_type}_{period}"
                paths[key] = filepath
    return paths


def extract_primary_document(filepath: str, doc_type: str = "10-K"):
    # raw SEC file ek container hai jisme sainkdo documents bunde hote hain
    # yahan sirf primary filing document nikalna hai
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", content, re.DOTALL)
    for doc in documents:
        type_match = re.search(r"<TYPE>(.*?)\n", doc)
        if type_match and type_match.group(1).strip() == doc_type:
            text_match = re.search(r"<TEXT>(.*)", doc, re.DOTALL)
            if text_match:
                return text_match.group(1)
    return None


def parse_sec_filing(filepath: str, doc_type: str):
    raw_html = extract_primary_document(filepath, doc_type)
    if raw_html is None:
        return None, []

    html_content = re.sub(r"</?XBRL>", "", raw_html)
    soup = BeautifulSoup(html_content, "lxml")

    # hidden inline-XBRL metadata hatao
    for hidden in soup.find_all("ix:header"):
        hidden.decompose()
    for hidden in soup.find_all(style=re.compile(r"display\s*:\s*none")):
        hidden.decompose()

    # tables alag nikalo, structured rows ke roop mein
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells_ = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells_):
                rows.append(cells_)
        if rows:
            tables.append(rows)
        table.decompose()

    plain_text = soup.get_text(separator="\n", strip=True)
    return plain_text, tables


def parse_all(filing_paths: dict) -> dict:
    """filing_paths: {doc_name: filepath}. Returns {doc_name: {"text": ..., "tables": [...]}}."""
    parsed = {}
    for name, path in filing_paths.items():
        doc_type = "10-K" if "10-K" in name else "10-Q"
        text, tables = parse_sec_filing(path, doc_type=doc_type)
        parsed[name] = {"text": text, "tables": tables}
        print(f"{name}: text={len(text) if text else 0} chars, tables={len(tables)}")
    return parsed


def build_company_filings(companies: dict, sec_email: str, download: bool = True) -> dict:
    """
    End-to-end: download (optional) + build paths + parse, for every company.
    Returns {company_display_name: {doc_name: {"text": ..., "tables": [...]}}}.
    """
    if download:
        download_filings(companies, sec_email)

    parsed_by_company = {}
    for name, ticker in companies.items():
        paths = build_filing_paths(ticker)
        print(f"{name} ({ticker}): {len(paths)} filings found")
        print(f"--- Parsing {name} ---")
        parsed_by_company[name] = parse_all(paths)
    return parsed_by_company
