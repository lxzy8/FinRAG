"""Word-based chunking with overlap, and context-prefixed chunk construction."""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list:
    # word-based chunking, overlap taaki chunk boundary pe sentence na kate
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def build_contextual_chunks(parsed: dict, company: str) -> list:
    """parsed: {doc_name: {"text": ..., "tables": [...]}} for a single company."""
    all_chunks = []
    for doc_name, data in parsed.items():
        doc_type, period = doc_name.split("_", 1)
        raw_chunks = chunk_text(data["text"])
        for i, chunk in enumerate(raw_chunks):
            # context prefix — company/doctype/period, taaki chunk apne aap mein meaningful rahe
            context_prefix = f"Company: {company}\nDocument: {doc_type}\nPeriod: {period}\n\n"
            all_chunks.append({
                "content": context_prefix + chunk,   # embedding ke liye (context ke saath)
                "raw_content": chunk,                 # BM25 aur citation ke liye (bina context)
                "company": company,
                "doc_type": doc_type,
                "period": period,
                "chunk_id": f"{doc_name}_chunk{i}",
            })
    return all_chunks


def build_all_chunks(parsed_by_company: dict) -> list:
    """parsed_by_company: {company: {doc_name: {"text":..., "tables":...}}}."""
    all_chunks = []
    for name, parsed in parsed_by_company.items():
        company_chunks = build_contextual_chunks(parsed, company=name)
        all_chunks.extend(company_chunks)
        print(f"{name} chunks: {len(company_chunks)}")
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks
