"""
ContextBuilder — turns evidence buckets into a structured context string.

Public API:
  builder = ContextBuilder()
  context_str = builder.build(evidence_buckets)
"""

from __future__ import annotations

import config
from src.models import ScoredChunk


# Rough chars-per-token ratio for budget estimation
_CHARS_PER_TOKEN = 4


class ContextBuilder:

    def deduplicate_chunks(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """
        Deduplicate chunks by doc_id, keeping the highest-scoring chunk
        per source document. This prevents multiple chunks from the same
        document consuming prompt budget and reduces redundancy.
        """
        seen: dict[str, ScoredChunk] = {}  # doc_id -> best chunk so far
        for chunk in chunks:
            doc_id = chunk.doc_id
            if doc_id not in seen or chunk.final_score > seen[doc_id].final_score:
                seen[doc_id] = chunk
        # return in score order
        return sorted(seen.values(), key=lambda c: -c.final_score)

    def apply_token_budget(
        self,
        chunks: list[ScoredChunk],
        max_tokens: int = config.MAX_CONTEXT_TOKENS,
    ) -> list[ScoredChunk]:
        """Trim chunks list so total text stays within approximate token budget."""
        budget_chars = max_tokens * _CHARS_PER_TOKEN
        result: list[ScoredChunk] = []
        used = 0
        for chunk in chunks:
            chunk_chars = len(chunk.text)
            if used + chunk_chars > budget_chars and result:
                break
            result.append(chunk)
            used += chunk_chars
        return result

    def _format_chunk(self, chunk: ScoredChunk) -> str:
        return f"[{chunk.doc_id}]\n{chunk.text}"

    def build(
        self,
        evidence_buckets: dict[str, list[ScoredChunk]],
    ) -> tuple[str, list[str]]:
        """
        Build a structured context string from evidence buckets.

        Returns:
            context_str : the formatted multi-section context block
            source_ids  : list of doc_ids represented in the context
        """
        all_chunks: list[ScoredChunk] = []
        for bucket_chunks in evidence_buckets.values():
            all_chunks.extend(bucket_chunks)

        all_chunks = self.deduplicate_chunks(all_chunks)
        all_chunks = self.apply_token_budget(all_chunks)

        # Rebuild buckets after dedup + budget trim
        trimmed_ids = {c.chunk_id for c in all_chunks}
        trimmed_buckets: dict[str, list[ScoredChunk]] = {
            bucket: [c for c in chunks if c.chunk_id in trimmed_ids]
            for bucket, chunks in evidence_buckets.items()
        }

        sections: list[str] = []
        source_ids: list[str] = []

        def _add_section(header: str, chunks: list[ScoredChunk]) -> None:
            if not chunks:
                return
            formatted = "\n\n".join(self._format_chunk(c) for c in chunks)
            sections.append(f"[{header}]\n{formatted}")
            for c in chunks:
                if c.doc_id not in source_ids:
                    source_ids.append(c.doc_id)

        _add_section("General Policy",           trimmed_buckets.get("general", []))
        _add_section("Program-Specific Policy",  trimmed_buckets.get("specific", []))
        _add_section("Exceptions / Overrides",   trimmed_buckets.get("exception", []))
        _add_section("Process / Next Steps",     trimmed_buckets.get("process", []))

        return "\n\n".join(sections), source_ids

