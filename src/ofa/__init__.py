"""
`ofa.pipeline.FinancialAnalystPipeline` is intentionally NOT imported here.

It pulls in langchain-google-genai, sentence-transformers, and faiss — heavy,
optional-at-test-time dependencies. Keeping this __init__.py minimal means
`ofa.extraction` and `ofa.agent` can be unit-tested (see tests/) without those
installed. Import the pipeline explicitly where you need it:

    from ofa.pipeline import FinancialAnalystPipeline
"""

__all__ = ["pipeline"]
