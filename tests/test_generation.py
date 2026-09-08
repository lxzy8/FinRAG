"""
Prompt formatting test — actual Qwen generation Kaggle pe test hoga
(llama.cpp server chahiye), yaha sirf prompt construction check kar rahe hain.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generation.prompts import format_rag_prompt, SYSTEM_PROMPT


def test_prompt_formatting():
    # Dummy data — real retrieval pipeline se nahi, sirf format check karne ke liye
    dummy_chunks = [
        {
            "text": "The Company's revenue for fiscal year 2024 was $391.0 billion...",
            "metadata": {
                "company": "AAPL",
                "form_type": "10-K",
                "section": "mda",
                "filing_date": "2024-11-01",
            },
        }
    ]

    dummy_xbrl = [
        {
            "concept": "Revenues",
            "value": 391035000000,
            "fiscal_year": 2024,
            "fiscal_period": "FY",
            "filed_date": "2024-11-01",
        }
    ]

    prompt = format_rag_prompt(
        query="What was Apple's revenue in FY2024?",
        retrieved_chunks=dummy_chunks,
        xbrl_facts=dummy_xbrl,
    )

    print("=== System Prompt ===")
    print(SYSTEM_PROMPT)
    print("\n=== User Prompt ===")
    print(prompt)


if __name__ == "__main__":
    test_prompt_formatting()