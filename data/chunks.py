"""
Section-based chunking for ABC RAG Assistant.

Splits each document into 2–4 semantic sections by double-newline paragraph
breaks. Each chunk inherits all metadata from the parent document, plus:
  chunk_id : "{doc_id}_c{n}"
  text     : the section text

Run directly to verify:
  python data/chunks.py
"""

from __future__ import annotations

import re
import sys
import os

# Allow running as script from any cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.documents import DOCUMENTS


def _split_sections(text: str, max_sections: int = 4) -> list[str]:
    """
    Split document text into semantic sections.

    Strategy:
    1. Split on double-newline paragraph breaks.
    2. Merge very short paragraphs (<80 chars) with the next one so every
       chunk has enough context.
    3. Cap at max_sections by merging trailing sections.
    """
    raw = re.split(r"\n{2,}", text.strip())
    raw = [p.strip() for p in raw if p.strip()]

    # Merge short paragraphs forward
    merged: list[str] = []
    buf = ""
    for para in raw:
        if buf:
            candidate = buf + "\n\n" + para
        else:
            candidate = para

        if len(buf) < 80:
            buf = candidate
        else:
            merged.append(buf)
            buf = para
    if buf:
        merged.append(buf)

    # Cap at max_sections
    if len(merged) <= max_sections:
        return merged

    # Merge excess sections into last chunk
    head = merged[: max_sections - 1]
    tail = "\n\n".join(merged[max_sections - 1 :])
    return head + [tail]


def build_chunks() -> list[dict]:
    """
    Return a flat list of chunk dicts for all 18 documents.

    Each chunk dict:
      chunk_id  : str       e.g. "gen_001_c1"
      doc_id    : str       parent document id
      text      : str       section text
      metadata  : dict      full metadata from parent doc (unchanged)
    """
    chunks: list[dict] = []
    for doc in DOCUMENTS:
        doc_id = doc["doc_id"]
        sections = _split_sections(doc["content"])
        for i, section in enumerate(sections, start=1):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_c{i}",
                    "doc_id": doc_id,
                    "text": section,
                    "metadata": {
                        **doc["metadata"],
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_c{i}",
                    },
                }
            )
    return chunks


# =============================================================================
# Verification — run: python data/chunks.py
# =============================================================================
if __name__ == "__main__":
    chunks = build_chunks()

    print(f"Total chunks: {len(chunks)}\n")
    print(f"{'chunk_id':<20} {'doc_id':<16} {'priority':<10} {'len':>6}  preview")
    print("-" * 90)

    current_doc = None
    for c in chunks:
        doc_id = c["doc_id"]
        if doc_id != current_doc:
            print()
            current_doc = doc_id
        preview = c["text"][:60].replace("\n", " ")
        print(
            f"{c['chunk_id']:<20} {doc_id:<16} "
            f"{c['metadata']['priority']:<10} {len(c['text']):>6}  {preview}..."
        )

    # Sanity checks
    doc_ids_in_chunks = {c["doc_id"] for c in chunks}
    doc_ids_in_docs = {d["doc_id"] for d in DOCUMENTS}
    missing = doc_ids_in_docs - doc_ids_in_chunks
    assert not missing, f"Documents with no chunks: {missing}"

    for c in chunks:
        assert len(c["text"]) >= 30, f"Chunk too short: {c['chunk_id']}"
        assert "doc_id" in c["metadata"]
        assert "priority" in c["metadata"]

    print(f"\nOK: All {len(DOCUMENTS)} documents chunked. {len(chunks)} total chunks. All checks passed.")
