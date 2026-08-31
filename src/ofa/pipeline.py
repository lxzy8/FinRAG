"""
Top-level orchestration. Typical usage:

    from ofa.pipeline import FinancialAnalystPipeline

    pipeline = FinancialAnalystPipeline()
    pipeline.ingest()        # downloads + parses SEC filings for DEFAULT_COMPANIES
    pipeline.build_index()   # chunks + embeds + builds hybrid retriever
    pipeline.compile()       # builds the LangGraph agent

    result = pipeline.run_query("What is AMD's revenue trend recently?")
    print(result["final_report"])
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from . import agent, config
from .chunking import build_all_chunks
from .extraction import LabelLearner
from .ingestion import build_company_filings
from .llm_utils import LLMStats
from .retrieval import HybridRetriever


class FinancialAnalystPipeline:
    def __init__(self, companies: dict = None, gemini_key: str = None, sec_email: str = None,
                 llm_model: str = config.DEFAULT_LLM_MODEL):
        self.companies = companies or dict(config.DEFAULT_COMPANIES)
        self.gemini_key = gemini_key or config.get_gemini_key()
        self.sec_email = sec_email or config.get_sec_email()

        self.llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=self.gemini_key)
        self.stats = LLMStats()
        self.label_learner = LabelLearner()

        self.parsed_by_company = {}
        self.all_chunks = []
        self.retriever = None
        self.nodes = None
        self.app = None

    # ---------- setup ----------

    def ingest(self, download: bool = True):
        """Download (optional) + parse SEC filings for every configured company."""
        self.parsed_by_company = build_company_filings(self.companies, self.sec_email, download=download)
        return self.parsed_by_company

    def build_index(self):
        """Chunk parsed filings and build the hybrid (dense + BM25) retriever."""
        if not self.parsed_by_company:
            raise RuntimeError("Call .ingest() before .build_index().")
        self.all_chunks = build_all_chunks(self.parsed_by_company)
        self.retriever = HybridRetriever(self.all_chunks)
        return self.all_chunks

    def compile(self):
        """Build the LangGraph agent nodes and compile the graph."""
        if self.retriever is None:
            raise RuntimeError("Call .build_index() before .compile().")
        self.nodes = agent.build_nodes(
            llm=self.llm, retriever=self.retriever, all_chunks=self.all_chunks,
            parsed_by_company=self.parsed_by_company, companies=self.companies,
            label_learner=self.label_learner, stats=self.stats,
        )
        self.app = agent.build_graph(self.nodes)
        return self.app

    def setup(self, download: bool = True):
        """Convenience: ingest + build_index + compile in one call."""
        self.ingest(download=download)
        self.build_index()
        self.compile()
        return self

    # ---------- querying ----------

    def run_query(self, query: str, reset_stats: bool = True) -> dict:
        if self.app is None:
            raise RuntimeError("Call .compile() (or .setup()) before .run_query().")
        if reset_stats:
            self.stats.reset()
        return self.app.invoke(agent.new_state(query))

    def naive_run_query(self, query: str, company: str) -> dict:
        """
        Minimal RAG baseline: single retrieval pass, no company filter, no
        deterministic calculation. Used by eval/run_eval.py for comparison.
        """
        from .llm_utils import extract_text, invoke_with_retry

        top_indices = self.retriever.search(query, top_k=10)
        evidence = [self.all_chunks[idx] for idx in top_indices]

        evidence_text = "\n\n".join(
            f"[{e['chunk_id']}] {e['raw_content'][:400]}" for e in evidence
        )
        prompt = f"""Question: {query}

Evidence:
{evidence_text}

Answer the question using the evidence above. Cite chunk_ids in [brackets] where relevant."""

        response = invoke_with_retry(self.llm, prompt, stats=self.stats)
        final_report = extract_text(response)
        return {"final_report": final_report, "evidence": evidence, "company": company}
