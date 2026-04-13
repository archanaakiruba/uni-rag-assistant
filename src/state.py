"""
ConversationState — in-memory CRUD for UserSession objects.

Public API:
  state = ConversationState()
  session = state.get_session(user_id)
  state.enrich_profile(session, intent)
  state.add_turn(user_id, question, answer, intent)
"""

from __future__ import annotations

import config
from src.models import IntentType, QueryIntent, UserProfile, UserSession


class ConversationState:

    def __init__(self) -> None:
        self._store: dict[str, UserSession] = {}

    def get_session(self, user_id: str) -> UserSession:
        """Return existing session or create a new one."""
        if user_id not in self._store:
            self._store[user_id] = UserSession()
        return self._store[user_id]

    def enrich_profile(self, session: UserSession, intent: QueryIntent) -> UserSession:
        """
        Merge detected intent signals into the stored user profile.
        Existing profile values are only overwritten when a new value is detected.
        """
        p = session.profile

        if intent.program:
            p.program = intent.program
        if intent.region:
            p.region = intent.region
        if intent.audience:
            p.audience = intent.audience
        if intent.intent != IntentType.UNKNOWN:
            p.last_intent = intent.intent

        # Infer audience from region
        if p.region == "non_eu" and not p.audience:
            p.audience = "international"

        # Infer scholarship interest from intent
        if intent.intent == IntentType.SCHOLARSHIP:
            p.needs_scholarship = True

        # Infer part-time interest from study mode intent
        if intent.intent == IntentType.STUDY_MODE:
            q_lower = intent.enriched_query.lower()
            if "part-time" in q_lower or "part time" in q_lower:
                p.wants_part_time = True
            elif "full-time" in q_lower or "full time" in q_lower:
                p.wants_part_time = False

        return session

    def add_turn(
        self,
        user_id: str,
        question: str,
        answer: str,
        intent: QueryIntent,
    ) -> None:
        """Append a Q/A turn to the session. Prune to MAX_TURNS."""
        session = self.get_session(user_id)
        session.turns.append({"q": question, "a": answer})
        if len(session.turns) > config.MAX_TURNS:
            session.turns = session.turns[-config.MAX_TURNS:]
        # Second enrich pass — captures any signals resolved during generation
        # (e.g. program confirmed from context). First pass is in app.py step 3.
        self.enrich_profile(session, intent)
