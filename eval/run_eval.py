"""
Evaluation suite — run with:

    python -m eval.run_eval

Requires GEMINI_API_KEY and SEC_EDGAR_EMAIL to be set as environment variables.
Runs the full pipeline (download -> parse -> index -> agent) once, then all
eval checks against it, and prints a summary.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ofa.agent import new_state
from ofa.pipeline import FinancialAnalystPipeline


TEST_QUESTIONS = {
    "NVIDIA": "Why did NVIDIA's gross margin change recently?",
    "AMD": "What is AMD's revenue trend recently?",
    "Apple": "What is Apple's revenue trend recently?",
    "Walmart": "How has Walmart's gross margin trended recently?",
    "Johnson & Johnson": "What is Johnson & Johnson's revenue trend recently?",
    "JPMorgan Chase": "What is JPMorgan Chase's revenue trend recently?",
}

TEST_QUERIES_FOR_CONTAMINATION = {
    "NVIDIA": ["What drove NVIDIA's gross margin change recently?", "How much did NVIDIA spend on R&D?"],
    "AMD": ["What is AMD's revenue trend recently?", "How much did AMD spend on R&D?"],
    "Apple": ["What is Apple's revenue trend recently?", "How much did Apple spend on R&D?"],
    "Walmart": ["What is Walmart's revenue trend recently?", "How has Walmart's gross margin changed?"],
    "Johnson & Johnson": ["What is Johnson & Johnson's revenue trend recently?", "How has JNJ's gross margin changed?"],
    "JPMorgan Chase": ["What is JPMorgan Chase's revenue trend recently?", "How has JPMorgan's revenue changed?"],
}


def eval_retrieval_contamination(pipeline, test_queries=TEST_QUERIES_FOR_CONTAMINATION):
    rows = []
    for company, queries in test_queries.items():
        for q in queries:
            state = pipeline.nodes["retrieve"](new_state(q, company=company))
            contaminated = [e["chunk_id"] for e in state["evidence"] if e["company"] != company]
            rows.append({"company": company, "query": q, "n_evidence": len(state["evidence"]),
                         "n_contaminated": len(contaminated)})

    total_evidence = sum(r["n_evidence"] for r in rows)
    total_contaminated = sum(r["n_contaminated"] for r in rows)
    rate = round(100 * total_contaminated / total_evidence, 2) if total_evidence else None
    print(f"Contamination rate: {rate}%  ({total_contaminated}/{total_evidence} chunks)")
    return {"rows": rows, "contamination_rate_pct": rate}


def eval_calculation_coverage(pipeline, results_by_company):
    summary = {}
    for company, filings in pipeline.parsed_by_company.items():
        calcs = results_by_company[company]["calculations"]
        calculated_docs = [k for k in calcs if not k.startswith("_")]
        total_docs = len(filings)
        coverage_pct = round(100 * len(calculated_docs) / total_docs, 1) if total_docs else None
        summary[company] = {"total_filings": total_docs, "calculated": len(calculated_docs),
                             "coverage_pct": coverage_pct, "missing_docs": [d for d in filings if d not in calculated_docs]}
        print(f"{company}: {len(calculated_docs)}/{total_docs} filings ({coverage_pct}%)")
    return summary


def eval_citation_groundedness(result):
    cited = set(re.findall(r"\[([\w\-]+_chunk\d+)\]", result["final_report"]))
    evidence_ids = {e["chunk_id"] for e in result["evidence"]}
    grounded = cited & evidence_ids
    groundedness_pct = round(100 * len(grounded) / len(cited), 1) if cited else None
    return {"n_cited": len(cited), "n_grounded": len(grounded), "groundedness_pct": groundedness_pct}


def eval_contamination_and_groundedness(result, expected_company):
    evidence_ids = {e["chunk_id"] for e in result["evidence"]}
    contaminated = [e["chunk_id"] for e in result["evidence"] if e["company"] != expected_company]
    cited = set(re.findall(r"\[([\w\-]+_chunk\d+)\]", result["final_report"]))
    grounded = cited & evidence_ids
    groundedness_pct = round(100 * len(grounded) / len(cited), 1) if cited else None
    return {"n_evidence": len(result["evidence"]), "n_contaminated": len(contaminated), "groundedness_pct": groundedness_pct}


def eval_naive_vs_agentic(pipeline, results_by_company, test_questions=TEST_QUESTIONS):
    rows = []
    for company, question in test_questions.items():
        naive_result = pipeline.naive_run_query(question, company)
        naive_metrics = eval_contamination_and_groundedness(naive_result, company)

        agentic_result = results_by_company[company]
        agentic_metrics = eval_contamination_and_groundedness(agentic_result, company)

        rows.append({
            "company": company,
            "naive_contamination": naive_metrics["n_contaminated"],
            "agentic_contamination": agentic_metrics["n_contaminated"],
            "naive_has_calc": "calculat" in naive_result["final_report"].lower(),
            "agentic_has_calc": bool(agentic_result.get("calculations")),
        })
        print(f"{company}: naive_contam={naive_metrics['n_contaminated']} "
              f"agentic_contam={agentic_metrics['n_contaminated']}")
    return rows


def eval_precision_recall_at_k(pipeline, k=10):
    rows = []
    for company, filings in pipeline.parsed_by_company.items():
        quarterly_docs = sorted([d for d in filings if d.startswith("10-Q")])
        if not quarterly_docs:
            continue
        target_doc = quarterly_docs[-1]

        query = f"What is {company}'s most recent quarterly revenue and gross margin?"
        state = pipeline.nodes["retrieve"](new_state(query, company=company))
        retrieved = state["evidence"][:k]

        relevant = [c for c in retrieved if c["chunk_id"].startswith(target_doc)]
        precision = round(len(relevant) / len(retrieved), 3) if retrieved else 0.0

        total_relevant = [c for c in pipeline.all_chunks
                           if c["company"] == company and c["chunk_id"].startswith(target_doc)]
        recall = round(len(relevant) / len(total_relevant), 3) if total_relevant else None

        rows.append({"company": company, "target_doc": target_doc, "precision_at_k": precision, "recall_at_k": recall})
        print(f"{company}: precision@{k}={precision}  recall@{k}={recall}")
    return rows


def eval_cost_latency(pipeline, test_questions=TEST_QUESTIONS):
    rows = []
    for company, question in test_questions.items():
        pipeline.stats.reset()
        start = time.time()
        result = pipeline.run_query(question, reset_stats=False)
        wall_time = round(time.time() - start, 2)
        rows.append({
            "company": company, "wall_time_sec": wall_time,
            "llm_calls": pipeline.stats.total_calls,
            "llm_time_sec": round(pipeline.stats.total_time, 2),
            "iterations": result["iteration"],
        })
        print(f"{company}: {wall_time}s wall, {pipeline.stats.total_calls} LLM calls, "
              f"{result['iteration']} iterations")
    return rows


def main():
    pipeline = FinancialAnalystPipeline()
    print("=== Setting up pipeline (download + parse + index + compile) ===")
    pipeline.setup(download=True)

    print("\n=== Running test queries ===")
    results_by_company = {c: pipeline.run_query(q) for c, q in TEST_QUESTIONS.items()}

    print("\n=== Retrieval contamination ===")
    contamination = eval_retrieval_contamination(pipeline)

    print("\n=== Calculation coverage ===")
    coverage = eval_calculation_coverage(pipeline, results_by_company)

    print("\n=== Citation groundedness ===")
    groundedness = {c: eval_citation_groundedness(r) for c, r in results_by_company.items()}
    for c, g in groundedness.items():
        print(f"{c}: {g['groundedness_pct']}%")

    print("\n=== Naive vs agentic baseline ===")
    baseline = eval_naive_vs_agentic(pipeline, results_by_company)

    print("\n=== Precision@10 / Recall@10 ===")
    precision_recall = eval_precision_recall_at_k(pipeline, k=10)

    print("\n=== Cost / latency ===")
    cost_latency = eval_cost_latency(pipeline)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Contamination rate: {contamination['contamination_rate_pct']}%")
    for c, s in coverage.items():
        print(f"{c} coverage: {s['coverage_pct']}%")
    for c, g in groundedness.items():
        print(f"{c} groundedness: {g['groundedness_pct']}%")


if __name__ == "__main__":
    main()
