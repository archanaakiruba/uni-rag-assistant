"""
One-time indexing script.

1. Loads all chunks from data/chunks.py
2. Embeds each chunk using OpenAI text-embedding-3-small
3. Stores embeddings + metadata in a persistent ChromaDB collection
4. Builds a BM25 index over all chunk texts and saves it to disk

Run with:
  python -m src.indexer
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from datetime import datetime

# Allow running as a script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

import config
from data.chunks import build_chunks


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


def _embed_batch(client: OpenAI, texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed a list of texts in batches, returning a flat list of vectors."""
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
        # Respect rate limits
        if start + batch_size < len(texts):
            time.sleep(0.5)
    return all_embeddings


def build_index(force: bool = False) -> None:
    """
    Embed all chunks and store in ChromaDB. Build BM25 index and save to disk.

    Args:
        force: If True, delete and rebuild the collection even if it exists.
    """
    if not config.OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

    # ------------------------------------------------------------------
    # ChromaDB setup
    # ------------------------------------------------------------------
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    if force:
        try:
            chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
            print(f"Deleted existing collection '{config.CHROMA_COLLECTION_NAME}'.")
        except Exception:
            pass

    existing_collections = [c.name for c in chroma_client.list_collections()]
    if config.CHROMA_COLLECTION_NAME in existing_collections and not force:
        collection = chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)
        count = collection.count()
        if count > 0:
            print(
                f"Collection '{config.CHROMA_COLLECTION_NAME}' already has {count} items. "
                "Skipping embedding. Use force=True to rebuild."
            )
            _ensure_bm25(force=False)
            return

    collection = chroma_client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ------------------------------------------------------------------
    # Load chunks
    # ------------------------------------------------------------------
    chunks = build_chunks()
    print(f"Loaded {len(chunks)} chunks from {len(set(c['doc_id'] for c in chunks))} documents.")

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    # ChromaDB metadata values must be str | int | float | bool.
    # Lists (e.g. topics) are joined as comma-separated strings so they can be
    # reconstructed in retriever.py via split(",") without ambiguity.
    def _sanitize_meta(m: dict) -> dict:
        sanitized = {}
        for k, v in m.items():
            if isinstance(v, list):
                sanitized[k] = ",".join(str(i) for i in v)  # e.g. "eligibility,transfer_credit"
            elif isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized

    metadatas = [_sanitize_meta(c["metadata"]) for c in chunks]

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    print(f"Embedding {len(texts)} chunks with {config.EMBEDDING_MODEL}...")
    embeddings = _embed_batch(openai_client, texts)
    print("Embedding complete.")

    # ------------------------------------------------------------------
    # Store in ChromaDB
    # ------------------------------------------------------------------
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Stored {len(ids)} chunks in ChromaDB collection '{config.CHROMA_COLLECTION_NAME}'.")

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------
    _ensure_bm25(chunks=chunks, force=True)

    print(
        f"\nIndex manifest: {len(chunks)} chunks | "
        f"{len(set(c['doc_id'] for c in chunks))} docs | "
        f"model: {config.EMBEDDING_MODEL} | "
        f"built: {datetime.now().isoformat(timespec='seconds')}"
    )


def _ensure_bm25(chunks: list[dict] | None = None, force: bool = False) -> None:
    """Build and save BM25 index if it doesn't exist or force=True."""
    if not force and os.path.exists(config.BM25_INDEX_PATH):
        print(f"BM25 index already exists at {config.BM25_INDEX_PATH}. Skipping.")
        return

    if chunks is None:
        chunks = build_chunks()

    tokenized = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    index_data = {
        "bm25": bm25,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "doc_ids": [c["doc_id"] for c in chunks],
        "texts": [c["text"] for c in chunks],
        "metadatas": [c["metadata"] for c in chunks],
    }

    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)

    print(f"BM25 index saved to {config.BM25_INDEX_PATH} ({len(chunks)} entries).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the ABC RAG index.")
    parser.add_argument(
        "--force", action="store_true", help="Delete and rebuild existing index."
    )
    args = parser.parse_args()

    build_index(force=args.force)
    print("Indexing complete.")
