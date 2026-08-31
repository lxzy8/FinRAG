"""
Unit tests for `ofa.extraction`. These use synthetic table data — no network,
no LLM, no SEC download — so they run fast and deterministically in CI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ofa.extraction import (
    LabelLearner,
    calc_cagr,
    calc_growth_rate,
    calc_margin,
    extract_metric_from_tables,
)


def make_parsed(tables):
    """Helper: wrap a list of tables into the {doc_name: {"tables": [...]}} shape."""
    return {"10-Q_2026-06-30": {"text": "", "tables": tables}}


def test_extract_metric_from_tables_basic_match():
    tables = [[["Total revenue", "$1,000", "$900"], ["Cost of revenue", "$400", "$350"]]]
    parsed = make_parsed(tables)

    revenue = extract_metric_from_tables(parsed, keywords=["total revenue"])
    cost = extract_metric_from_tables(parsed, keywords=["cost of revenue"])

    assert revenue["10-Q_2026-06-30"][0]["values"] == [1000.0, 900.0]
    assert cost["10-Q_2026-06-30"][0]["values"] == [400.0, 350.0]


def test_extract_metric_from_tables_handles_negative_parentheses():
    tables = [[["Net loss", "($250)", "$100"]]]
    parsed = make_parsed(tables)

    result = extract_metric_from_tables(parsed, keywords=["net loss"])

    assert result["10-Q_2026-06-30"][0]["values"] == [-250.0, 100.0]


def test_extract_metric_from_tables_no_match_returns_empty():
    tables = [[["Sales to customers", "$1,000"]]]
    parsed = make_parsed(tables)

    # AMD/NVIDIA-style keyword won't match a J&J-style label
    result = extract_metric_from_tables(parsed, keywords=["total revenue", "net sales"])

    assert result == {}


def test_extract_metric_from_tables_tracks_table_idx():
    tables = [
        [["Some unrelated row", "1"]],
        [["Total revenue", "$500"]],
    ]
    parsed = make_parsed(tables)

    result = extract_metric_from_tables(parsed, keywords=["total revenue"])

    assert result["10-Q_2026-06-30"][0]["table_idx"] == 1


class _FakeLLMResponse:
    def __init__(self, text):
        self.content = text


class _FakeLLM:
    """Always classifies the first row label as the answer — enough to test the fallback wiring."""

    def __init__(self, label_to_return):
        self.label_to_return = label_to_return
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return _FakeLLMResponse('{"label": "%s"}' % self.label_to_return)


def test_label_learner_fallback_learns_and_caches():
    tables = [[["Sales to customers", "$1,000"]]]
    parsed = make_parsed(tables)

    fake_llm = _FakeLLM(label_to_return="sales to customers")
    learner = LabelLearner()

    # first call: keyword miss -> LLM fallback triggers, learns the label
    result1 = learner.extract_with_fallback(
        fake_llm, parsed, base_keywords=["total revenue"], metric_type="revenue", company="Test Co",
    )
    assert "sales to customers" in learner.learned_keywords["revenue"]
    assert fake_llm.calls == 1
    assert result1["10-Q_2026-06-30"][0]["values"] == [1000.0]

    # second call, same doc: should NOT call the LLM again (cached)
    learner.extract_with_fallback(
        fake_llm, parsed, base_keywords=["total revenue"], metric_type="revenue", company="Test Co",
    )
    assert fake_llm.calls == 1  # unchanged


def test_calc_margin():
    assert calc_margin(1000, 400) == 60.0
    assert calc_margin(0, 400) is None


def test_calc_growth_rate():
    assert calc_growth_rate(100, 150) == 50.0
    assert calc_growth_rate(0, 150) is None


def test_calc_cagr():
    assert calc_cagr(100, 121, 2) == 10.0
    assert calc_cagr(0, 121, 2) is None
