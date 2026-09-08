"""
SEC EDGAR se structured XBRL numeric data (revenue, EPS, etc.) nikaalna.
"""

import requests
import json
from pathlib import Path

HEADERS = {
    "User-Agent": "FinRAG Project your-email@example.com"  # apna email daal dena
}

XBRL_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def get_xbrl_facts(cik: str) -> dict:
    """Company ka poora structured XBRL data laata hai."""
    url = XBRL_FACTS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def extract_metric(xbrl_data: dict, concept: str, unit: str = "USD") -> list[dict]:
    """
    XBRL data se ek specific metric nikaalta hai
    (e.g. 'Revenues', 'EarningsPerShareDiluted').
    """
    try:
        facts = xbrl_data["facts"]["us-gaap"][concept]["units"][unit]
    except KeyError:
        return []

    results = []
    for entry in facts:
        results.append({
            "value": entry["val"],
            "fiscal_year": entry.get("fy"),
            "fiscal_period": entry.get("fp"),
            "start_date": entry.get("start"),
            "end_date": entry.get("end"),
            "form": entry.get("form"),
            "filed_date": entry.get("filed"),
        })

    return results


def save_json(data: dict, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)