"""
Rule-based intent + program + region extraction, with async LLM fallback
for UNKNOWN intents.

parse(question, session) -> QueryIntent   (async)
"""

from __future__ import annotations

from openai import AsyncOpenAI

import config
from src.models import IntentType, QueryIntent, UserSession


# ---------------------------------------------------------------------------
# Keyword rule tables (from SYSTEM_DESIGN.md §5.3)
# ---------------------------------------------------------------------------

INTENT_RULES: dict[IntentType, list[str]] = {
    IntentType.ELIGIBILITY:     ["eligible", "can i apply", "qualify", "requirements", "accepted", "background"],
    IntentType.SCHOLARSHIP:     ["scholarship", "financial aid", "funding", "bursary", "merit"],
    IntentType.NEXT_STEPS:      ["next steps", "what should i do", "what should i do next", "how do i", "process", "apply now"],
    IntentType.STUDY_MODE:      ["part-time", "full-time", "while working", "online", "campus", "flexible"],
    IntentType.LANGUAGE_REQ:    ["ielts", "toefl", "english", "language", "proficiency"],
    IntentType.TRANSFER_CREDIT: ["transfer", "credit", "recognition", "prior", "previous degree"],
    IntentType.TUITION:         ["cost", "tuition", "fee", "payment", "monthly", "afford"],
    IntentType.COMPARE_OPTIONS: ["difference between", "compare", "vs", "versus", "better", "which program"],
    IntentType.VISA:            ["visa", "non-eu", "india", "international", "immigration"],
}

PROGRAM_RULES: dict[str, list[str]] = {
    "msc_ds": ["data science", "msc data", "msc ds"],
    "msc_ai": ["applied ai", "msc ai", "artificial intelligence master", "msc applied ai"],
    "mba":    ["mba", "business administration master", "master of business"],
    "bsc_cs": ["computer science bachelor", "bsc cs", "bsc computer", "bachelor computer science"],
    "bsc_ba": ["business administration bachelor", "bsc ba", "bsc business"],
}

REGION_RULES: dict[str, list[str]] = {
    "non_eu": [
        "india", "indian", "china", "nigeria", "nigerian", "pakistan",
        "non-eu", "non eu", "outside europe", "outside the eu",
        "international student", "visa required",
    ],
    "eu": ["eu student", "european", "germany", "german", "austria", "france"],
}

AUDIENCE_RULES: dict[str, list[str]] = {
    "working_professional": ["work experience", "years of experience", "employed", "working", "job", "professional experience"],
    "international":        ["international", "from abroad", "non-eu", "visa", "nigeria", "india", "china", "pakistan"],
    "career_changer":       ["career change", "career changer", "switching field", "different background", "new field"],
    "postgraduate":         ["master", "msc", "mba", "postgrad"],
}

EXCEPTION_HINTS: list[str] = [
    "no degree", "without a degree", "work experience", "years of experience",
    "non-eu", "india", "nigeria", "conditional", "pending", "ielts", "toefl",
    "credit transfer", "transfer credits", "exception", "override",
    "scholarship", "merit", "mba", "without certificate",
]

PROCESS_HINTS: list[str] = [
    "next steps", "what should i do", "how do i apply", "process", "timeline",
    "when", "how long", "start date", "apply", "deadline", "steps",
]

DEGREE_KEYWORDS: dict[str, str] = {
    "economics":        "economics",
    "business":         "business",
    "computer science": "computer_science",
    "engineering":      "engineering",
    "mathematics":      "mathematics",
    "maths":            "mathematics",
    "statistics":       "statistics",
    "finance":          "finance",
    "social science":   "social_science",
    "humanities":       "humanities",
    "psychology":       "psychology",
}

WORK_EXP_PATTERNS: list[str] = [
    "years of experience", "years experience", "work experience",
    "years working", "professional experience", "i work", "i am working",
    "currently working", "full-time job", "employed",
]

LANG_PENDING_PATTERNS: list[str] = [
    "waiting for ielts", "ielts pending", "ielts result",
    "toefl pending", "toefl result", "language test pending",
    "haven't taken", "not yet taken", "pending language",
    "before i get my ielts", "before my ielts",
]

# Map LLM label strings to IntentType enum values
_INTENT_LABEL_MAP: dict[str, IntentType] = {
    it.value: it for it in IntentType
}

_openai_client: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


# ---------------------------------------------------------------------------
# LLM fallback for UNKNOWN intents
# ---------------------------------------------------------------------------

async def llm_classify_intent(question: str, session: UserSession) -> IntentType:
    """
    Cheap GPT-4o-mini call to classify an intent that rule-based matching missed.
    Only called when the rule-based pass returns UNKNOWN.
    Never raises — falls back to UNKNOWN on any error.
    """
    valid_labels = ", ".join(it.value for it in IntentType)
    system_prompt = (
        "Classify the user question into exactly one of these intents: "
        f"{valid_labels}. "
        "Return only the intent label, nothing else."
    )
    try:
        client = _get_openai()
        response = await client.chat.completions.create(
            model=config.LLM_INTENT_FALLBACK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=20,
        )
        label = (response.choices[0].message.content or "").strip().lower()
        return _INTENT_LABEL_MAP.get(label, IntentType.UNKNOWN)
    except Exception:
        return IntentType.UNKNOWN


# ---------------------------------------------------------------------------
# Pure helpers (sync)
# ---------------------------------------------------------------------------

def _lower(text: str) -> str:
    return text.lower()


def _detect_intent(q: str) -> IntentType:
    q_lower = _lower(q)
    scores: dict[IntentType, int] = {t: 0 for t in IntentType}
    for intent_type, keywords in INTENT_RULES.items():
        for kw in keywords:
            if kw in q_lower:
                scores[intent_type] += 1
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 0 else IntentType.UNKNOWN


def _detect_program(q: str) -> str | None:
    q_lower = _lower(q)
    for program, keywords in PROGRAM_RULES.items():
        for kw in keywords:
            if kw in q_lower:
                return program
    return None


def _detect_region(q: str) -> str | None:
    q_lower = _lower(q)
    for region, keywords in REGION_RULES.items():
        for kw in keywords:
            if kw in q_lower:
                return region
    return None


def _detect_audience(q: str) -> str | None:
    q_lower = _lower(q)
    for audience, keywords in AUDIENCE_RULES.items():
        for kw in keywords:
            if kw in q_lower:
                return audience
    return None


def _needs_exception(q: str, intent: IntentType) -> bool:
    q_lower = _lower(q)
    if intent in (IntentType.SCHOLARSHIP, IntentType.VISA, IntentType.TRANSFER_CREDIT):
        return True
    return any(hint in q_lower for hint in EXCEPTION_HINTS)


def _needs_process(q: str, intent: IntentType) -> bool:
    q_lower = _lower(q)
    if intent == IntentType.NEXT_STEPS:
        return True
    return any(hint in q_lower for hint in PROCESS_HINTS)


def _build_enriched_query(
    question: str,
    intent: IntentType,
    program: str | None,
    region: str | None,
    audience: str | None,
    session: UserSession,
) -> str:
    """Combine question with resolved profile context into a richer retrieval query."""
    parts = [question]

    profile = session.profile
    effective_program = program or profile.program
    effective_region = region or profile.region
    effective_audience = audience or profile.audience

    if effective_program:
        parts.append(f"program: {effective_program}")
    if effective_region:
        parts.append(f"region: {effective_region}")
    if effective_audience:
        parts.append(f"audience: {effective_audience}")
    if intent != IntentType.UNKNOWN:
        parts.append(f"intent: {intent.value}")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------

async def parse(question: str, session: UserSession) -> QueryIntent:
    """
    Parse a question and session state into a QueryIntent.

    Rule-based extraction runs first (zero latency). If it returns UNKNOWN,
    a cheap GPT-4o-mini call classifies the intent. LLM failure falls back
    to UNKNOWN — the pipeline continues either way.

    For compare_options intent: intent.program is set to None regardless of
    which programs are mentioned, so ChromaDB searches the full corpus and
    both program docs can be retrieved.
    """
    intent_type = _detect_intent(question)
    program = _detect_program(question)
    region = _detect_region(question)
    audience = _detect_audience(question)

    # Extract additional profile signals directly into session.profile
    question_lower = _lower(question)

    # 2a. Degree background (first match wins; never overwrite existing value)
    if session.profile.highest_degree is None:
        for keyword, degree_value in DEGREE_KEYWORDS.items():
            if keyword in question_lower:
                session.profile.highest_degree = degree_value
                break

    # 2b. Work experience mention
    if any(p in question_lower for p in WORK_EXP_PATTERNS):
        session.profile.work_experience_mentioned = True

    # 2c. Language proof pending
    if any(p in question_lower for p in LANG_PENDING_PATTERNS):
        session.profile.language_proof_status = "pending"

    # Inherit from profile where not detected in the current question
    profile = session.profile
    program = program or profile.program
    region = region or profile.region
    audience = audience or profile.audience

    # Detect compare_options: set program=None so retrieval searches full corpus
    if intent_type == IntentType.COMPARE_OPTIONS or "difference between" in question.lower():
        intent_type = IntentType.COMPARE_OPTIONS
        program = None  # must not filter to a single program

    # LLM fallback — only for UNKNOWN after rule-based pass
    if intent_type == IntentType.UNKNOWN:
        intent_type = await llm_classify_intent(question, session)
        # If LLM resolves to compare_options, also clear program
        if intent_type == IntentType.COMPARE_OPTIONS:
            program = None

    enriched_query = _build_enriched_query(question, intent_type, program, region, audience, session)

    # Non-EU region always implies exception docs are relevant (exc_001, faq_002),
    # even when the current question doesn't use exception-hint keywords.
    # This matters most for follow-up turns where region is inherited from profile.
    needs_exception = _needs_exception(question, intent_type) or (region == "non_eu")

    return QueryIntent(
        intent=intent_type,
        program=program,
        region=region,
        audience=audience,
        needs_exception=needs_exception,
        needs_process=_needs_process(question, intent_type),
        enriched_query=enriched_query,
    )
