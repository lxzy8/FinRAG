"""
Full evaluation runner — retrieval metrics, numeric accuracy, aur latency.
Generation quality (faithfulness, relevancy) ke liye RAGAS Kaggle pe
chalega jab actual Qwen model available hoga (yaha structure hi hai).
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.golden_qa import load_golden_qa
from eval.metrics import (
    precision_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    check_numeric_accuracy,
)


def run_retrieval_eval(hybrid_index, reranker_fn, top_k: int = 8) -> dict:
    """
    Golden QA set ke har question pe retrieval chalata hai, aur
    precision/MRR/hit-rate measure karta hai.
    """
    questions = load_golden_qa()
    results = []

    for q in questions:
        if q["relevant_chunk_metadata"] is None:
            continue  # web-search-only questions skip karo retrieval eval mein

        start = time.time()
        candidates = hybrid_index.hybrid_search(q["question"], top_k=50)
        retrieved = reranker_fn(q["question"], candidates, top_k=top_k)
        latency = time.time() - start

        precision = precision_at_k(retrieved, q["relevant_chunk_metadata"], k=top_k)
        mrr = mean_reciprocal_rank(retrieved, q["relevant_chunk_metadata"])
        hit = hit_rate_at_k(retrieved, q["relevant_chunk_metadata"], k=top_k)

        results.append({
            "question_id": q["id"],
            "precision_at_k": precision,
            "mrr": mrr,
            "hit_rate": hit,
            "latency_sec": latency,
        })

    avg_precision = sum(r["precision_at_k"] for r in results) / len(results) if results else 0
    avg_mrr = sum(r["mrr"] for r in results) / len(results) if results else 0
    hit_rate = sum(r["hit_rate"] for r in results) / len(results) if results else 0
    avg_latency = sum(r["latency_sec"] for r in results) / len(results) if results else 0

    return {
        "per_question": results,
        "avg_precision_at_k": avg_precision,
        "avg_mrr": avg_mrr,
        "hit_rate": hit_rate,
        "avg_retrieval_latency_sec": avg_latency,
    }


def run_numeric_accuracy_eval(generate_fn) -> dict:
    """
    Numeric questions pe full pipeline chalata hai, aur answer mein
    sahi number aaya ya nahi check karta hai (XBRL ground truth se).

    generate_fn: function jo question leke final answer string deta hai
    (poori RAG pipeline ke through — retrieval + generation dono)
    """
    from eval.golden_qa import get_numeric_questions

    questions = get_numeric_questions()
    results = []

    for q in questions:
        answer = generate_fn(q["question"])
        check = check_numeric_accuracy(answer, q["expected_numeric"])

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "generated_answer": answer,
            **check,
        })

    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0

    return {
        "per_question": results,
        "numeric_accuracy": accuracy,
    }


def print_eval_report(retrieval_results: dict, numeric_results: dict = None):
    """Human-readable summary print karta hai."""
    print("=" * 50)
    print("RETRIEVAL EVALUATION")
    print("=" * 50)
    print(f"Avg Precision@k: {retrieval_results['avg_precision_at_k']:.3f}")
    print(f"Avg MRR: {retrieval_results['avg_mrr']:.3f}")
    print(f"Hit Rate: {retrieval_results['hit_rate']:.3f}")
    print(f"Avg Retrieval Latency: {retrieval_results['avg_retrieval_latency_sec']:.3f}s")

    if numeric_results:
        print("\n" + "=" * 50)
        print("NUMERIC ACCURACY EVALUATION")
        print("=" * 50)
        print(f"Numeric Accuracy: {numeric_results['numeric_accuracy']:.1%}")
        for r in numeric_results["per_question"]:
            status = "✓" if r["correct"] else "✗"
            print(f"  {status} {r['question_id']}: expected {r['expected']}, found {r['found_numbers']}")