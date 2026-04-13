"""
Assembles the final system + user prompt dicts for GPT-4o.

Public API:
  builder = PromptBuilder()
  messages = builder.build(question, context_str, session)
"""

from __future__ import annotations

import config
from src.models import UserSession

SYSTEM_PROMPT = """\
You are a study advisory assistant for ABC University.

Rules:
1. Answer ONLY using the context provided below.
2. When general policy and program-specific or exception policy conflict,
   the more specific or exception document takes precedence.
3. If the answer depends on information the user has not provided
   (e.g. their region, degree background), you MUST ask a short clarifying
   question instead of assuming or listing multiple alternative cases.
   Do not present multiple conditional scenarios without first asking which
   applies. Example: if asked "Can I apply?" without degree info, respond
   with "That depends on your degree background — what is your Bachelor's
   degree in?" rather than listing all possible cases.
4. If retrieved context is insufficient to answer confidently, say:
   "I don't have enough information in my current knowledge to answer
   that reliably. You may want to contact ABC admissions directly."
5. Never invent policies, dates, or fees.
6. End every answer by listing the source document IDs you used, on a line
   starting with "Sources:" followed by comma-separated IDs.
"""

USER_PROMPT_TEMPLATE = """\
Question: {question}

Resolved user context:
{user_context_block}

Retrieved context:
{retrieved_context}

Conversation history:
{history_block}
"""


class PromptBuilder:

    def format_user_context(self, session: UserSession) -> str:
        p = session.profile
        lines: list[str] = []
        if p.program:
            lines.append(f"- Program of interest: {p.program}")
        if p.region:
            lines.append(f"- Region: {p.region}")
        if p.audience:
            lines.append(f"- Audience type: {p.audience}")
        if p.highest_degree:
            lines.append(f"- Highest degree: {p.highest_degree}")
        if p.wants_part_time is not None:
            lines.append(f"- Wants part-time: {p.wants_part_time}")
        if p.needs_scholarship is not None:
            lines.append(f"- Needs scholarship: {p.needs_scholarship}")
        if p.language_proof_status:
            lines.append(f"- Language proof status: {p.language_proof_status}")
        if p.work_experience_mentioned:
            lines.append("- Work experience mentioned: Yes")
        if p.last_intent:
            lines.append(f"- Last intent: {p.last_intent.value}")
        return "\n".join(lines) if lines else "(none yet)"

    def format_history(self, session: UserSession) -> str:
        turns = session.turns[-config.HISTORY_TURNS:]
        if not turns:
            return "(no prior turns)"
        lines: list[str] = []
        for i, turn in enumerate(turns, start=1):
            lines.append(f"Turn {i}:")
            lines.append(f"  Q: {turn.get('q', '')}")
            lines.append(f"  A: {turn.get('a', '')}")
        return "\n".join(lines)

    def build(
        self,
        question: str,
        context_str: str,
        session: UserSession,
    ) -> list[dict[str, str]]:
        """
        Returns a list of messages dicts for the OpenAI chat completion API.
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        user_context_block = self.format_user_context(session)
        history_block = self.format_history(session)

        user_content = (
            USER_PROMPT_TEMPLATE
            .replace("{question}", question)
            .replace("{user_context_block}", user_context_block)
            .replace("{retrieved_context}", context_str)
            .replace("{history_block}", history_block)
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
