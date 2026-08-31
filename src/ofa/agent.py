"""
LangGraph agent: contextualize -> plan -> retrieve <-> verify -> calculate -> synthesize.

Nodes are built by `build_nodes(...)`, which closes over the shared pipeline
state (LLM, retriever, parsed filings, label learner) rather than relying on
notebook-style module globals. `build_graph(...)` wires those nodes into a
compiled LangGraph app.
"""

from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from .extraction import calc_growth_rate, calc_margin
from .llm_utils import extract_text, invoke_with_retry
from .retrieval import detect_recency_intent, detect_doc_type_intent, get_recent_target_docs


class AgentState(TypedDict):
    query: str                   # user ka sawaal
    company: str                  # active entity
    sub_queries: List[str]        # Plan node ke sub-questions
    evidence: List[dict]          # Retrieve node ke chunks
    verification_notes: str       # Verify node ka verdict
    iteration: int                # retry counter
    calculations: dict            # Calculate node ka output
    final_report: str             # Synthesize node ka final answer


def new_state(query: str, company: str = "") -> AgentState:
    return {
        "query": query, "company": company, "sub_queries": [], "evidence": [],
        "verification_notes": "", "iteration": 0, "calculations": {}, "final_report": "",
    }


def build_nodes(llm, retriever, all_chunks, parsed_by_company, companies, label_learner, stats=None):
    """
    Returns a dict of node callables: {"contextualize": fn, "plan": fn, ...}.
    Each closes over the pipeline state passed in here — no module-level globals.
    """

    def contextualize_node(state: AgentState) -> AgentState:
        print("-> Contextualize node running...")
        # keyword-match company naam ya ticker se — abhi simple, sophisticated
        # entity-resolution (NER/embedding-match) baad mein
        query_lower = state["query"].lower()
        detected = None
        for name, ticker in companies.items():
            if name.lower() in query_lower or ticker.lower() in query_lower:
                detected = name
                break
        state["company"] = detected or next(iter(companies))  # default: pehli company
        return state

    def plan_node(state: AgentState) -> AgentState:
        print("-> Plan node running...")
        prompt = f"""Given this financial research question about {state['company']}, break it into 2-4 specific, focused sub-questions that would help research the answer. Return ONLY the sub-questions, one per line, no numbering, no extra text.

Question: {state['query']}"""
        response = invoke_with_retry(llm, prompt, stats=stats)
        text = extract_text(response)
        state["sub_queries"] = [line.strip() for line in text.split("\n") if line.strip()]
        return state

    def retrieve_node(state: AgentState) -> AgentState:
        print("-> Retrieve node running...")
        state["iteration"] += 1  # yahan increment karo (asli node hai, iska return persist hota hai)

        seen_chunk_ids = {e["chunk_id"] for e in state.get("evidence", [])}
        all_evidence = state.get("evidence", [])
        queries_to_search = state["sub_queries"] if state["sub_queries"] else [state["query"]]

        # Pre-filter: pehle candidate pool banao (company, +period agar recency-intent
        # hai), phir usi pool ke andar ranking karo — post-filter (global top-k, phir
        # discard) mein slots irrelevant companies/periods pe waste ho jaate hain.
        recency = detect_recency_intent(state["query"])
        doc_type_intent = detect_doc_type_intent(state["query"]) if recency else None
        target_docs = (
            get_recent_target_docs(parsed_by_company, state["company"], doc_type_filter=doc_type_intent)
            if recency else None
        )
        target_prefixes = tuple(target_docs.values()) if target_docs else None

        candidate_indices = [
            i for i, c in enumerate(all_chunks)
            if c["company"] == state["company"]
            and (not target_prefixes or c["chunk_id"].startswith(target_prefixes))
        ]
        # Fallback: company ke paas target period ki filing hi na ho (rare), to
        # sirf company-filter tak wapas aa jao — kabhi khaali candidate pool na ho
        if recency and not candidate_indices:
            candidate_indices = [i for i, c in enumerate(all_chunks) if c["company"] == state["company"]]

        for sq in queries_to_search:
            for idx in retriever.search_within(sq, candidate_indices, top_k=15):
                chunk = all_chunks[idx]
                if chunk["chunk_id"] not in seen_chunk_ids:
                    all_evidence.append(chunk)
                    seen_chunk_ids.add(chunk["chunk_id"])

        state["evidence"] = all_evidence
        return state

    def verify_node(state: AgentState) -> AgentState:
        print("-> Verify node running...")
        evidence_text = "\n\n".join(
            f"[{e['chunk_id']}] {e['raw_content'][:300]}" for e in state["evidence"][:8]
        )
        prompt = f"""Question: {state['query']}

Evidence collected so far:
{evidence_text}

Is this evidence sufficient to answer the question well? Reply with exactly one word first (SUFFICIENT or INSUFFICIENT), then a brief one-line reason."""
        response = invoke_with_retry(llm, prompt, stats=stats)
        state["verification_notes"] = extract_text(response)
        return state

    def should_continue_research(state: AgentState) -> str:
        # sirf read karta hai, mutate NAHI karta (routing function ka mutation persist nahi hota)
        if state["iteration"] >= 3:
            return "calculate"
        if "INSUFFICIENT" in state["verification_notes"].upper():
            return "retrieve"
        return "calculate"

    def calculate_node(state: AgentState) -> AgentState:
        print("-> Calculate node running...")
        company_filings = parsed_by_company.get(state["company"], {})

        # pehle static keyword-matching try hota hai; agar kisi filing ke liye
        # kuch match nahi hota, LLM fallback us doc ke labels classify karta hai
        # (ek baar; result learned_keywords mein permanently cache ho jaata hai)
        revenue_data = label_learner.extract_with_fallback(
            llm, company_filings, base_keywords=["total revenue", "revenue", "net revenue", "net sales"],
            metric_type="revenue", company=state["company"], stats=stats,
        )
        cost_data = label_learner.extract_with_fallback(
            llm, company_filings, base_keywords=["cost of revenue", "cost of sales", "cost of products sold"],
            metric_type="cost", company=state["company"], stats=stats,
        )

        calculations = {}
        for doc_name in revenue_data:
            if doc_name not in cost_data:
                print(f"No cost-of-revenue match for {doc_name} — check filing's actual label wording")
                continue

            # Same-table match dhoondo, warna nearby tables allow karo (window=2)
            # — kuch filings (especially 10-K) mein ek hi income statement
            # multiple adjacent <table> tags mein split hoti hai
            matched, best_gap = None, None
            for rev_row in revenue_data[doc_name]:
                for cost_row in cost_data[doc_name]:
                    gap = abs(rev_row["table_idx"] - cost_row["table_idx"])
                    if gap <= 2 and (best_gap is None or gap < best_gap):
                        matched, best_gap = (rev_row, cost_row), gap

            if not matched:
                print(f"No nearby revenue/cost pair for {doc_name} (checked window=2)")
                continue

            rev_row, cost_row = matched
            if rev_row["values"] and cost_row["values"]:
                revenue = rev_row["values"][0]
                cost = cost_row["values"][0]
                # sanity check: cost, revenue ka 10%-95% ke beech hona chahiye
                # (isse bahar hai matlab galat row match hua, koi ratio/percentage row)
                if revenue > 0 and (0.10 * revenue <= cost <= 0.95 * revenue):
                    calculations[doc_name] = {
                        "revenue": revenue,
                        "cost_of_revenue": cost,
                        "gross_margin_pct": calc_margin(revenue, cost),
                    }
                else:
                    print(f"Skipped {doc_name}: cost={cost}, revenue={revenue}")  # sanity check fail

        quarterly_docs = sorted([d for d in calculations if d.startswith("10-Q")])
        if len(quarterly_docs) >= 2:
            oldest, newest = quarterly_docs[0], quarterly_docs[-1]
            growth = calc_growth_rate(calculations[oldest]["revenue"], calculations[newest]["revenue"])
            calculations["_quarterly_revenue_growth"] = {"from": oldest, "to": newest, "growth_pct": growth}

        state["calculations"] = calculations
        return state

    def synthesize_node(state: AgentState) -> AgentState:
        print("-> Synthesize node running...")
        evidence_text = "\n\n".join(
            f"[{e['chunk_id']}] {e['raw_content'][:400]}" for e in state["evidence"][:8]
        )

        calc_text = ""
        if state["calculations"]:
            calc_lines = []
            for k, v in state["calculations"].items():
                if not k.startswith("_"):
                    calc_lines.append(f"{k}: revenue=${v['revenue']}M, gross_margin={v['gross_margin_pct']}%")
                else:
                    calc_lines.append(f"{k}: {v}")
            calc_text = "Deterministically calculated financial metrics:\n" + "\n".join(calc_lines)

        prompt = f"""You are a financial research analyst. Answer the following question using the evidence and calculations provided.

IMPORTANT: You MUST explicitly cite the specific numeric values from "Deterministically calculated financial metrics" section below (exact percentages, dollar figures) in your answer, not just qualitative statements. Cite chunk IDs in brackets like [chunk_id] when referencing text evidence.

Question: {state['query']}

Evidence:
{evidence_text}

{calc_text}

Write a clear, well-organized answer that includes the specific calculated numbers."""
        response = invoke_with_retry(llm, prompt, stats=stats)
        state["final_report"] = extract_text(response)
        return state

    return {
        "contextualize": contextualize_node,
        "plan": plan_node,
        "retrieve": retrieve_node,
        "verify": verify_node,
        "should_continue_research": should_continue_research,
        "calculate": calculate_node,
        "synthesize": synthesize_node,
    }


def build_graph(nodes: dict):
    graph = StateGraph(AgentState)
    graph.add_node("contextualize", nodes["contextualize"])
    graph.add_node("plan", nodes["plan"])
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("verify", nodes["verify"])
    graph.add_node("calculate", nodes["calculate"])
    graph.add_node("synthesize", nodes["synthesize"])

    graph.set_entry_point("contextualize")
    graph.add_edge("contextualize", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "verify")
    graph.add_conditional_edges(
        "verify", nodes["should_continue_research"], {"retrieve": "retrieve", "calculate": "calculate"}
    )
    graph.add_edge("calculate", "synthesize")
    graph.add_edge("synthesize", END)

    app = graph.compile()
    print("Graph compiled")
    return app
