"""
All dataclasses and Pydantic schemas — no logic.

Contains:
  IntentType      — enum of recognised query intents
  QueryIntent     — structured result of intent parsing
  UserProfile     — accumulated user context across turns
  UserSession     — per-user conversation state
  ScoredChunk     — retrieval result with score
  AskRequest      — Pydantic request schema for POST /ask
  AskResponse     — Pydantic response schema for POST /ask
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    ELIGIBILITY     = "eligibility"
    SCHOLARSHIP     = "scholarship"
    NEXT_STEPS      = "next_steps"
    STUDY_MODE      = "study_mode"
    LANGUAGE_REQ    = "language_requirement"
    TRANSFER_CREDIT = "transfer_credit"
    TUITION         = "tuition"
    COMPARE_OPTIONS = "compare_options"
    VISA            = "visa"
    UNKNOWN         = "unknown"


# ---------------------------------------------------------------------------
# Query intent — output of intent_parser
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    intent: IntentType
    program: str | None           # "msc_ds", "mba", …  or None
    region: str | None            # "non_eu", "eu"       or None
    audience: str | None          # "working_professional", "international", …
    needs_exception: bool
    needs_process: bool
    enriched_query: str           # question + resolved profile context


# ---------------------------------------------------------------------------
# User profile — accumulated across turns
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    program: str | None                 = None  # last discussed program
    region: str | None                  = None  # "eu" | "non_eu"
    audience: str | None                = None  # "working_professional" etc.
    highest_degree: str | None              = None
    wants_part_time: bool | None            = None
    needs_scholarship: bool | None          = None
    language_proof_status: str | None       = None  # "has_cert" | "pending" | None
    work_experience_mentioned: bool | None  = None
    last_intent: IntentType | None          = None


# ---------------------------------------------------------------------------
# User session — per-user conversation state
# ---------------------------------------------------------------------------

@dataclass
class UserSession:
    turns: list[dict[str, Any]] = field(default_factory=list)   # [{"q":…,"a":…}, …] max 5
    profile: UserProfile        = field(default_factory=UserProfile)


# ---------------------------------------------------------------------------
# Scored chunk — retrieval result
# ---------------------------------------------------------------------------

@dataclass
class ScoredChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]
    rrf_score: float    = 0.0   # score after RRF fusion
    final_score: float  = 0.0   # score after metadata boost


# ---------------------------------------------------------------------------
# API schemas (Pydantic v2)
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    user_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
