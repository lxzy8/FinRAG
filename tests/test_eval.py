"""
Metrics logic ka smoke test — dummy data se, taaki formulas sahi
kaam kar rahe hain confirm ho jaye. Full pipeline eval Kaggle pe hoga.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.metrics import (
    precision_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    check_numeric_accuracy,
    extract_numbers_from_text,
)


def test_retrieval_metrics():
    dummy_chunks = [
        {"metadata": {"company": "AAPL", "section": "risk_factors"}},
        {"metadata": {"company": "AAPL", "section": "mda"}},
        {"metadata": {"company": "AAPL", "section": "risk_factors"}},
        {"metadata": {"company": "MSFT", "section": "risk_factors"}},  # irrelevant
    ]
    target = {"company": "AAPL", "section": "risk_factors"}

    precision = precision_at_k(dummy_chunks, target, k=4)
    mrr = mean_reciprocal_rank(dummy_chunks, target)
    hit = hit_rate_at_k(dummy_chunks, target, k=4)

    print(f"✓ Precision@4: {precision}")  # expect 0.5 (2 out of 4 match)
    print(f"✓ MRR: {mrr}")                 # expect 1.0 (first chunk matches)
    print(f"✓ Hit rate: {hit}")            # expect True


def test_numeric_accuracy():
    answer = "Apple's revenue for FY2024 was approximately $391.0 billion."
    expected = {"value": 391035000000, "tolerance_pct": 0.5}

    numbers = extract_numbers_from_text(answer)
    print(f"✓ Extracted numbers: {numbers}")

    result = check_numeric_accuracy(answer, expected)
    print(f"✓ Numeric check: {result}")


if __name__ == "__main__":
    print("=== Testing retrieval metrics ===")
    test_retrieval_metrics()

    print("\n=== Testing numeric accuracy ===")
    test_numeric_accuracy()