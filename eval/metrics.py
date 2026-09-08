"""
Custom metrics — Precision/Recall@k for retrieval, aur numeric
accuracy check (XBRL ground truth ke against).
"""


def precision_at_k(retrieved_chunks: list[dict], relevant_metadata: dict, k: int = 8) -> float:
    """
    Retrieved chunks mein se kitne percent actually relevant hain,
    metadata match ke basis pe (company, form_type, section).
    """
    if not retrieved_chunks:
        return 0.0

    top_k = retrieved_chunks[:k]
    relevant_count = sum(1 for c in top_k if _matches_metadata(c["metadata"], relevant_metadata))

    return relevant_count / len(top_k)


def recall_at_k(
    retrieved_chunks: list[dict],
    all_relevant_chunk_ids: list[str],
    k: int = 8,
) -> float:
    """
    Corpus mein jitne bhi relevant chunks the, unme se kitne percent
    top-k mein retrieve ho gaye.
    """
    if not all_relevant_chunk_ids:
        return None  # measure nahi kar sakte agar ground truth hi nahi hai

    top_k_ids = {c["chunk_id"] for c in retrieved_chunks[:k]}
    relevant_ids = set(all_relevant_chunk_ids)

    found = len(top_k_ids & relevant_ids)
    return found / len(relevant_ids)


def _matches_metadata(chunk_meta: dict, target_meta: dict) -> bool:
    """Helper — check karta hai ki chunk ka metadata target se match karta hai ya nahi."""
    for key, value in target_meta.items():
        if chunk_meta.get(key) != value:
            return False
    return True


def mean_reciprocal_rank(retrieved_chunks: list[dict], relevant_metadata: dict) -> float:
    """
    Pehla relevant chunk kis position pe mila — 1/position.
    Agar koi relevant chunk nahi mila, 0 return karta hai.
    """
    for i, chunk in enumerate(retrieved_chunks):
        if _matches_metadata(chunk["metadata"], relevant_metadata):
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(retrieved_chunks: list[dict], relevant_metadata: dict, k: int = 8) -> bool:
    """Kya top-k mein kam se kam ek relevant chunk aaya ya nahi (binary)."""
    top_k = retrieved_chunks[:k]
    return any(_matches_metadata(c["metadata"], relevant_metadata) for c in top_k)


def extract_numbers_from_text(text: str) -> list[float]:
    """Generated answer text se numbers nikaalta hai (billions/millions handle karte hue)."""
    import re

    # Patterns: "$391.0 billion", "391,035,000,000", "391.0B", etc.
    numbers = []

    # $X billion / $X million pattern
    for match in re.finditer(r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)?", text, re.IGNORECASE):
        num_str, unit = match.groups()
        try:
            num = float(num_str.replace(",", ""))
            if unit and unit.lower() in ("billion", "b"):
                num *= 1_000_000_000
            elif unit and unit.lower() in ("million", "m"):
                num *= 1_000_000
            numbers.append(num)
        except ValueError:
            continue

    return numbers


def check_numeric_accuracy(generated_answer: str, expected_numeric: dict) -> dict:
    """
    Generated answer mein expected number hai ya nahi, given tolerance.
    Returns: {"correct": bool, "found_numbers": [...], "expected": ...}
    """
    expected_value = expected_numeric["value"]
    tolerance_pct = expected_numeric.get("tolerance_pct", 0.1)
    tolerance = expected_value * (tolerance_pct / 100)

    found_numbers = extract_numbers_from_text(generated_answer)

    correct = any(
        abs(num - expected_value) <= tolerance for num in found_numbers
    )

    return {
        "correct": correct,
        "found_numbers": found_numbers,
        "expected": expected_value,
    }