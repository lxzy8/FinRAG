"""
Golden QA dataset — manually curated questions with ground-truth
answers, relevant chunk IDs (for retrieval eval), aur expected
numeric values (XBRL ground truth se, numeric accuracy check ke liye).

NOTE: Ye sirf 3 sample entries hain structure dikhane ke liye —
asli set 50-100 questions ka banega, apne corpus pe based.
"""

GOLDEN_QA = [
    {
        "id": "q001",
        "question": "What was Apple's total revenue for fiscal year 2024?",
        "expected_answer": "Apple's total revenue for FY2024 was approximately $391.0 billion.",
        "expected_numeric": {
            "value": 391035000000,
            "tolerance_pct": 0.1,  # ±0.1% tolerance for rounding
        },
        "relevant_chunk_metadata": {
            "company": "AAPL",
            "form_type": "10-K",
            "section": "mda",  # ya financial_statements
        },
        "requires_web_search": False,
        "category": "numeric_factual",
    },
    {
        "id": "q002",
        "question": "What risk factors did Apple mention related to supply chain?",
        "expected_answer": None,  # narrative — exact match nahi, judge-based eval hoga
        "expected_numeric": None,
        "relevant_chunk_metadata": {
            "company": "AAPL",
            "form_type": "10-K",
            "section": "risk_factors",
        },
        "requires_web_search": False,
        "category": "narrative_qualitative",
    },
    {
        "id": "q003",
        "question": "What is Apple's current stock price?",
        "expected_answer": None,  # live data, filing se nahi milega
        "expected_numeric": None,
        "relevant_chunk_metadata": None,
        "requires_web_search": True,
        "category": "live_info",
    },
]


def load_golden_qa() -> list[dict]:
    """Golden QA set return karta hai. Baad mein isse JSON file se load kar sakte ho."""
    return GOLDEN_QA


def get_numeric_questions() -> list[dict]:
    """Sirf wo questions jinke liye numeric ground truth hai."""
    return [q for q in GOLDEN_QA if q.get("expected_numeric")]


def get_by_category(category: str) -> list[dict]:
    """Category-wise filter (numeric_factual, narrative_qualitative, live_info)."""
    return [q for q in GOLDEN_QA if q["category"] == category]