"""
Prompt templates — RAG context ko Qwen ke liye structured prompt mein
format karna, XBRL numeric data ko bhi include karna.
"""

SYSTEM_PROMPT = """You are FinRAG, a financial research assistant. You answer questions \
strictly based on the provided context from SEC filings and structured financial data.

Rules:
- If the context does not contain enough information to answer, say so clearly. Do not guess.
- When citing numbers, prefer the structured XBRL data over narrative text if both are present.
- Always mention which filing (company, form type, fiscal year) your answer is based on.
- Be precise and concise. Avoid speculation or investment advice."""


def format_rag_prompt(
    query: str,
    retrieved_chunks: list[dict],
    xbrl_facts: list[dict] | None = None,
) -> str:
    """
    Retrieved chunks + XBRL facts ko ek structured prompt mein combine karta hai.
    """
    context_parts = []

    if xbrl_facts:
        context_parts.append("=== Structured Financial Data (XBRL) ===")
        for fact in xbrl_facts:
            context_parts.append(
                f"- {fact.get('concept', 'Value')}: {fact['value']} "
                f"(FY{fact.get('fiscal_year')}, {fact.get('fiscal_period')}, "
                f"filed {fact.get('filed_date')})"
            )
        context_parts.append("")

    context_parts.append("=== Retrieved Filing Excerpts ===")
    for i, chunk in enumerate(retrieved_chunks):
        meta = chunk["metadata"]
        context_parts.append(
            f"\n[Excerpt {i+1} — {meta['company']} {meta['form_type']}, "
            f"section: {meta['section']}, filed {meta['filing_date']}]"
        )
        context_parts.append(chunk["text"])

    context_str = "\n".join(context_parts)

    return f"""Context:
{context_str}

Question: {query}

Answer the question based only on the context above."""