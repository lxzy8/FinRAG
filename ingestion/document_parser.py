"""
Raw filing HTML se clean narrative text nikaalna,
aur usko standard sections mein todna (MD&A, Risk Factors, etc.)
"""

import re
from bs4 import BeautifulSoup

SECTION_PATTERNS = {
    "risk_factors": r"item\s*1a\.?\s*risk factors",
    "properties": r"item\s*2\.?\s*properties",
    "legal_proceedings": r"item\s*3\.?\s*legal proceedings",
    "mda": r"item\s*7\.?\s*management.?s discussion and analysis",
    "market_risk": r"item\s*7a\.?\s*quantitative and qualitative disclosures",
    "financial_statements": r"item\s*8\.?\s*financial statements",
    "controls_procedures": r"item\s*9a\.?\s*controls and procedures",
}


def clean_html_to_text(html: str) -> str:
    """Raw HTML se clean text nikaalta hai."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_into_sections(text: str) -> dict[str, str]:
    """Poore filing text ko named sections mein todta hai."""
    text_lower = text.lower()

    matches = []
    for section_name, pattern in SECTION_PATTERNS.items():
        for match in re.finditer(pattern, text_lower):
            matches.append((match.start(), section_name))

    matches.sort(key=lambda x: x[0])

    if not matches:
        return {"full_text": text}

    sections = {}
    for i, (start_pos, name) in enumerate(matches):
        end_pos = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        section_text = text[start_pos:end_pos].strip()

        if name not in sections or len(section_text) > len(sections[name]):
            sections[name] = section_text

    return sections


def parse_filing(html: str) -> dict:
    """Main entry point — raw HTML se clean, sectioned text deta hai."""
    clean_text = clean_html_to_text(html)
    sections = split_into_sections(clean_text)

    return {
        "full_text_length": len(clean_text),
        "sections": sections,
        "sections_found": list(sections.keys()),
    }