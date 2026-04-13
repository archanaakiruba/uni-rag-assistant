"""
Guardrails — confidence check, scope detection, fallback responses.

Public API:
  confidence_too_low(evidence_buckets) -> bool
  is_out_of_scope(intent) -> bool
"""

from __future__ import annotations

import config
from src.models import IntentType, QueryIntent, ScoredChunk


# Intents that are always in-scope
IN_SCOPE_INTENTS: set[IntentType] = {
    IntentType.ELIGIBILITY,
    IntentType.SCHOLARSHIP,
    IntentType.NEXT_STEPS,
    IntentType.STUDY_MODE,
    IntentType.LANGUAGE_REQ,
    IntentType.TRANSFER_CREDIT,
    IntentType.TUITION,
    IntentType.COMPARE_OPTIONS,
    IntentType.VISA,
}

# Keywords that strongly suggest an out-of-scope query
OUT_OF_SCOPE_KEYWORDS: list[str] = [
    "weather", "recipe", "sport", "football", "stock", "investment", "political", 
    "election", "relationship", "medical advice", "legal advice", "tax advice", "news",
]


def confidence_too_low(
    evidence_buckets: dict[str, list[ScoredChunk]],
    intent: QueryIntent | None = None,
) -> bool:
    """Return True if no chunk clears the confidence threshold.

    compare_options uses a lower bar (0.05) because scores are naturally
    diluted across multiple programs when no metadata pre-filter is applied.
    """
    all_chunks = [c for chunks in evidence_buckets.values() for c in chunks]
    if not all_chunks:
        return True
    threshold = (
        0.05
        if intent is not None and intent.intent == IntentType.COMPARE_OPTIONS
        else config.RETRIEVAL_CONFIDENCE_THRESHOLD
    )
    max_score = max(c.final_score for c in all_chunks)
    return max_score < threshold


def is_out_of_scope(intent: QueryIntent) -> bool:
    """Return True if the query is clearly outside ABC advisory topics."""
    if intent.intent in IN_SCOPE_INTENTS:
        return False

    q_lower = intent.enriched_query.lower()
    if any(kw in q_lower for kw in OUT_OF_SCOPE_KEYWORDS):
        return True

    # UNKNOWN intent with no program/region signals = probably off-topic
    if (
        intent.intent == IntentType.UNKNOWN
        and not intent.program
        and not intent.region
        and not intent.needs_exception
        and not intent.needs_process
    ):
        return True

    return False
