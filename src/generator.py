"""
LLMGenerator — calls GPT-4o and extracts sources from the answer.

Public API:
  gen = LLMGenerator()
  answer, sources = await gen.generate(messages, candidate_source_ids)
"""

from __future__ import annotations

import re

from openai import AsyncOpenAI

import config


class LLMGenerator:

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    def extract_sources_from_answer(
        self,
        answer: str,
        candidate_source_ids: list[str],
    ) -> list[str]:
        """
        Extract doc_ids from the answer's "Sources:" line.
        Falls back to returning all candidate_source_ids if parsing fails.
        """
        match = re.search(r"sources?:\s*(.+)", answer, re.IGNORECASE)
        if not match:
            return candidate_source_ids

        raw = match.group(1)
        tokens = re.findall(r"[a-z_0-9]+", raw.lower())
        # O(1) lookup per token instead of O(S) nested loop
        sid_map = {sid.lower(): sid for sid in candidate_source_ids}
        found: list[str] = []
        for token in tokens:
            if token in sid_map and sid_map[token] not in found:
                found.append(sid_map[token])

        return found if found else candidate_source_ids

    async def generate(
        self,
        messages: list[dict[str, str]],
        candidate_source_ids: list[str],
    ) -> tuple[str, list[str]]:
        """
        Call GPT-4o and return (answer_text, source_ids).

        answer_text has the "Sources:" line stripped for cleanliness.
        source_ids  is the list of doc_ids extracted from the answer.
        """
        response = await self._client.chat.completions.create(
            model=config.LLM_GENERATION_MODEL,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        full_answer: str = response.choices[0].message.content or ""

        sources = self.extract_sources_from_answer(full_answer, candidate_source_ids)

        # Strip the Sources line from the displayed answer
        clean_answer = re.sub(r"\n?sources?:.*$", "", full_answer, flags=re.IGNORECASE | re.DOTALL).strip()

        # If the LLM signals it couldn't answer, sources are misleading — suppress them
        _insufficient_signals = ("don't have enough", "do not have enough", "not enough information", "cannot answer")
        if any(sig in clean_answer.lower() for sig in _insufficient_signals):
            sources = []

        return clean_answer, sources
