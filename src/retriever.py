"""
HybridRetriever — dense + sparse retrieval with RRF fusion and metadata re-ranking.

Public API:
  retriever = HybridRetriever()
  evidence  = await retriever.retrieve(intent)  -> dict[str, list[ScoredChunk]]
"""

from __future__ import annotations

import asyncio
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

import config
from src.models import QueryIntent, ScoredChunk


# Map doc_type values (from metadata) to bucket names used in context builder.
# Assignment is always static — intent drives retrieval/scoring, not labeling.
DOC_TYPE_TO_BUCKET: dict[str, str] = {
    "general":      "general",
    "program":      "specific",
    "exception":    "exception",
    "faq":          "process",
    "guidance":     "specific",
}


class HybridRetriever:
    """
    Loads ChromaDB collection and BM25 index on first use (lazy init).
    Call await retrieve(intent) to get ranked evidence buckets.
    ChromaDB and BM25 calls are offloaded to a thread executor so they
    never block the asyncio event loop.
    """

    def __init__(self) -> None:
        self._chroma: chromadb.Collection | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_chunk_ids: list[str] = []
        self._bm25_doc_ids: list[str] = []
        self._bm25_texts: list[str] = []
        self._bm25_metadatas: list[dict] = []
        self._openai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    # ------------------------------------------------------------------
    # Lazy loading (sync — called once, from executor)
    # ------------------------------------------------------------------

    def _load_chroma(self) -> chromadb.Collection:
        if self._chroma is None:
            client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            self._chroma = client.get_collection(config.CHROMA_COLLECTION_NAME)
        return self._chroma

    def _load_bm25(self) -> None:
        if self._bm25 is not None:
            return
        if not os.path.exists(config.BM25_INDEX_PATH):
            raise FileNotFoundError(
                f"BM25 index not found at {config.BM25_INDEX_PATH}. "
                "Run: python -m src.indexer"
            )
        with open(config.BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._bm25_chunk_ids = data["chunk_ids"]
        self._bm25_doc_ids = data["doc_ids"]
        self._bm25_texts = data["texts"]
        self._bm25_metadatas = data["metadatas"]

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    async def dense_retrieve(self, intent: QueryIntent) -> list[ScoredChunk]:
        """Query ChromaDB with metadata pre-filter. Non-blocking via executor."""
        # Embed the enriched query (async OpenAI call)
        response = await self._openai.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=intent.enriched_query,
        )
        query_embedding = response.data[0].embedding

        # Metadata pre-filter runs BEFORE similarity scoring.
        # ChromaDB first filters the candidate pool by metadata, then ranks
        # within that pool by vector similarity.
        #
        # The "$in" clause always includes "all" so that general policy documents
        # (which have program="all" and region="all") are never excluded.
        # Only other programs' specific docs are filtered out.
        #
        # Exception for compare_options intent: intent.program is left None
        # so no program filter is applied and the full corpus is searched.
        # This is required when the user is comparing two programs side by side.

        filters: list[dict] = []
        if intent.program:
            filters.append({"program": {"$in": [intent.program, "all"]}})
        if intent.region:
            filters.append({"region": {"$in": [intent.region, "all"]}})

        # Build where clause — ChromaDB requires exactly one top-level operator.
        # Multiple conditions must be wrapped in $and; a single condition must
        # NOT be wrapped ({"$and": [single]} is also rejected).
        
        if len(filters) == 0:
            where_clause: dict | None = None
        elif len(filters) == 1:
            where_clause = filters[0]
        else:
            where_clause = {"$and": filters}

        loop = asyncio.get_running_loop()

        def _query() -> dict:
            collection = self._load_chroma()
            kwargs: dict = {
                "query_embeddings": [query_embedding],
                "n_results": config.DENSE_TOP_K,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_clause:
                kwargs["where"] = where_clause
            return collection.query(**kwargs)

        results = await loop.run_in_executor(None, _query)

        chunks: list[ScoredChunk] = []
        for i, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB returns cosine distance; convert to similarity
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            chunks.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    doc_id=results["metadatas"][0][i].get("doc_id", chunk_id.rsplit("_c", 1)[0]),
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    rrf_score=similarity,
                    final_score=similarity,
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Sparse retrieval
    # ------------------------------------------------------------------

    async def sparse_retrieve(self, intent: QueryIntent) -> list[ScoredChunk]:
        """BM25 retrieval over all chunks. Offloaded to executor for consistency."""
        loop = asyncio.get_running_loop()

        def _score() -> list[ScoredChunk]:
            self._load_bm25()
            assert self._bm25 is not None
            query_tokens = intent.enriched_query.lower().split()
            scores = self._bm25.get_scores(query_tokens)
            # argpartition is O(N) for top-k selection vs O(N log N) for full argsort.
            # Only the final k winners are sorted — O(k log k), k<<N.
            k = config.SPARSE_TOP_K
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            result: list[ScoredChunk] = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                result.append(
                    ScoredChunk(
                        chunk_id=self._bm25_chunk_ids[idx],
                        doc_id=self._bm25_doc_ids[idx],
                        text=self._bm25_texts[idx],
                        metadata=self._bm25_metadatas[idx],
                        rrf_score=float(scores[idx]),
                        final_score=float(scores[idx]),
                    )
                )
            return result

        return await loop.run_in_executor(None, _score)

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def rrf_fusion(
        dense: list[ScoredChunk],
        sparse: list[ScoredChunk],
        k: int = config.RRF_K,
    ) -> list[ScoredChunk]:
        """
        Reciprocal Rank Fusion over dense and sparse ranked lists.
        RRF score = sum(1 / (k + rank_i)) across lists.
        Returns merged, sorted list.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_index: dict[str, ScoredChunk] = {}

        for rank, chunk in enumerate(dense, start=1):
            rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)
            chunk_index[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse, start=1):
            rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)
            chunk_index[chunk.chunk_id] = chunk

        merged: list[ScoredChunk] = []
        for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
            c = chunk_index[chunk_id]
            c.rrf_score = score
            c.final_score = score
            merged.append(c)

        return merged

    # ------------------------------------------------------------------
    # Metadata scoring boost (§5.4)
    # ------------------------------------------------------------------

    @staticmethod
    def score_and_rerank(
        chunks: list[ScoredChunk],
        intent: QueryIntent,
    ) -> list[ScoredChunk]:
        """Apply metadata scoring boosts then sort descending."""
        for chunk in chunks:
            score = chunk.rrf_score
            meta = chunk.metadata

            if intent.program and meta.get("program") in (intent.program, "all"):
                score += 0.15
            topics_raw = meta.get("topics", "")
            topics_list = topics_raw.split(",") if isinstance(topics_raw, str) else topics_raw
            if intent.intent.value in topics_list:
                score += 0.10
            if intent.needs_exception and meta.get("priority") == 3:
                score += 0.12
            if intent.region and meta.get("region") in (intent.region, "all"):
                score += 0.08
            if intent.audience and meta.get("audience") in (intent.audience, "all"):
                score += 0.06
            if meta.get("doc_type") == "guidance":
                if intent.audience and meta.get("audience") == intent.audience:
                    score += 0.10  # exact audience match — same weight as topic match
                else:
                    score += 0.05  # guidance docs are generally useful for personalised answers

            chunk.final_score = score

        return sorted(chunks, key=lambda c: -c.final_score)

    # ------------------------------------------------------------------
    # Evidence bucket assembly (§5.5)
    # ------------------------------------------------------------------

    @staticmethod
    def assemble_evidence_buckets(
        chunks: list[ScoredChunk],
        intent: QueryIntent,
        top_k: int = config.FINAL_TOP_K,
    ) -> dict[str, list[ScoredChunk]]:
        """
        Select top_k chunks then distribute into named buckets based solely on
        each chunk's doc_type. Intent drives retrieval and scoring — it does not
        change how a retrieved chunk is labeled in the context block.

        Buckets: general | specific | exception | process
        """
        selected = chunks[:top_k]

        buckets: dict[str, list[ScoredChunk]] = {
            "general": [],
            "specific": [],
            "exception": [],
            "process": [],
        }

        for chunk in selected:
            doc_type = chunk.metadata.get("doc_type", "general")
            bucket = DOC_TYPE_TO_BUCKET.get(doc_type, "general")
            buckets[bucket].append(chunk)

        return buckets

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def retrieve(self, intent: QueryIntent) -> dict[str, list[ScoredChunk]]:
        """
        Full hybrid retrieval pipeline. Fully async — safe to await in FastAPI.

        Returns evidence buckets: dict[bucket_name, list[ScoredChunk]]
        """
        dense, sparse = await asyncio.gather(
            self.dense_retrieve(intent),
            self.sparse_retrieve(intent),
        )
        fused = self.rrf_fusion(dense, sparse)
        reranked = self.score_and_rerank(fused, intent)
        buckets = self.assemble_evidence_buckets(reranked, intent)
        return buckets
