"""
Unit tests for `ofa.agent` nodes. Uses a fake retriever (no real embeddings/FAISS)
so these run fast and without any model downloads — they test the *filtering
logic* (company scoping, recency pre-filter), not retrieval ranking quality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ofa.agent import build_nodes, new_state
from ofa.extraction import LabelLearner


class _FakeRetriever:
    """search_within just returns the candidate indices as-is (no real ranking)."""

    def search_within(self, query, candidate_indices, top_k=10):
        return list(candidate_indices)[:top_k]

    def search(self, query, top_k=10):
        raise NotImplementedError("not used by retrieve_node")


def _make_chunks():
    chunks = []
    for company, doc_names in [
        ("NVIDIA", ["10-Q_2026-07-26", "10-K_2026-01-25"]),
        ("AMD", ["10-Q_2026-06-27", "10-K_2025-12-27"]),
    ]:
        for doc_name in doc_names:
            for i in range(3):
                chunks.append({
                    "content": f"{company} {doc_name} chunk {i}",
                    "raw_content": f"{company} {doc_name} chunk {i}",
                    "company": company,
                    "doc_type": doc_name.split("_")[0],
                    "period": doc_name.split("_")[1],
                    "chunk_id": f"{doc_name}_chunk{i}",
                })
    return chunks


def _build_test_nodes():
    all_chunks = _make_chunks()
    parsed_by_company = {
        "NVIDIA": {"10-Q_2026-07-26": {}, "10-K_2026-01-25": {}},
        "AMD": {"10-Q_2026-06-27": {}, "10-K_2025-12-27": {}},
    }
    companies = {"NVIDIA": "NVDA", "AMD": "AMD"}

    nodes = build_nodes(
        llm=None,  # retrieve_node doesn't call the LLM
        retriever=_FakeRetriever(),
        all_chunks=all_chunks,
        parsed_by_company=parsed_by_company,
        companies=companies,
        label_learner=LabelLearner(),
    )
    return nodes, all_chunks


def test_retrieve_node_never_returns_wrong_company():
    nodes, _ = _build_test_nodes()
    state = new_state("What is AMD's revenue trend?", company="AMD")

    result = nodes["retrieve"](state)

    assert len(result["evidence"]) > 0
    assert all(e["company"] == "AMD" for e in result["evidence"])


def test_retrieve_node_recency_prefilter_restricts_to_latest_quarter():
    nodes, _ = _build_test_nodes()
    # "recently" + "quarterly" -> should restrict candidates to the latest 10-Q only
    state = new_state("What is AMD's most recent quarterly revenue?", company="AMD")

    result = nodes["retrieve"](state)

    doc_names_retrieved = {e["chunk_id"].rsplit("_chunk", 1)[0] for e in result["evidence"]}
    assert doc_names_retrieved == {"10-Q_2026-06-27"}


def test_should_continue_research_stops_after_three_iterations():
    nodes, _ = _build_test_nodes()
    state = new_state("test query", company="AMD")
    state["iteration"] = 3
    state["verification_notes"] = "INSUFFICIENT evidence"

    assert nodes["should_continue_research"](state) == "calculate"


def test_should_continue_research_retries_when_insufficient():
    nodes, _ = _build_test_nodes()
    state = new_state("test query", company="AMD")
    state["iteration"] = 1
    state["verification_notes"] = "INSUFFICIENT — not enough evidence"

    assert nodes["should_continue_research"](state) == "retrieve"
