"""
Sectioned filing text ko RAG-ready chunks mein todna,
metadata (company, fiscal year, section) ke saath tag karna.
"""

import re
import uuid


def split_into_paragraphs(text: str) -> list[str]:
    """Text ko paragraphs mein todta hai (blank line se separate)."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Text ko fixed-size (character-based) overlapping chunks mein todta hai,
    lekin paragraph boundaries ka khayal rakhta hai — beech paragraph
    mein cut nahi karta jab tak zaroori na ho.
    """
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # Agar current chunk mein ye paragraph add karne se limit cross ho jaye
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap ke liye pichle chunk ka last hissa naye chunk mein le aao
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + para
        else:
            current_chunk += ("\n\n" if current_chunk else "") + para

        # Agar akela paragraph hi chunk_size se bada hai, usko bhi todna padega
        while len(current_chunk) > chunk_size * 1.5:
            chunks.append(current_chunk[:chunk_size].strip())
            current_chunk = current_chunk[chunk_size - chunk_overlap:]

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def create_chunks_with_metadata(
    sections: dict[str, str],
    company_ticker: str,
    fiscal_year: str,
    form_type: str,
    filing_date: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """
    Saare sections ko chunk karke, har chunk ke saath metadata attach karta hai.
    Ye final output hai jo embeddings/vector DB mein jayega.
    """
    all_chunks = []

    for section_name, section_text in sections.items():
        text_chunks = chunk_text(section_text, chunk_size, chunk_overlap)

        for idx, chunk in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {
                    "company": company_ticker,
                    "fiscal_year": fiscal_year,
                    "form_type": form_type,
                    "filing_date": filing_date,
                    "section": section_name,
                    "chunk_index_in_section": idx,
                },
            })

    return all_chunks


def print_chunk_stats(chunks: list[dict]):
    """Debugging ke liye — chunk sizes aur section distribution dekhne ke liye."""
    if not chunks:
        print("No chunks created.")
        return

    lengths = [len(c["text"]) for c in chunks]
    sections = {}
    for c in chunks:
        sec = c["metadata"]["section"]
        sections[sec] = sections.get(sec, 0) + 1

    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk length: {sum(lengths) / len(lengths):.0f} chars")
    print(f"Min/Max chunk length: {min(lengths)} / {max(lengths)}")
    print(f"Chunks per section: {sections}")