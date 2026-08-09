from __future__ import annotations

import json

from ..llm import LLM
from .models import (
    ExperienceEntry,
    SummaryEntry,
    KnowledgeEntry,
)


class Summary:

    def __init__(
        self,
        llm: LLM,
    ):
        self.llm = llm

    async def summarize(
        self,
        experiences: list[ExperienceEntry],
        knowledge: list[KnowledgeEntry],
    ) -> SummaryEntry:

        prompt = f"""
You are V.

Experiences:

{experiences}

Current knowledge:

{knowledge}

Create a concise summary.

Return ONLY valid JSON.

Schema:

{{
    "summary": "...",
    "lessons": [
        "...",
        "..."
    ],
    "confidence": 0.0,
    "importance": "low | medium | high"
}}

Rules:

- Merge duplicated lessons.
- Keep only reusable knowledge.
- Ignore temporary observations.
- Confidence must be between 0.0 and 1.0.
- Output ONLY JSON.
- No markdown.
""".strip()

        response = await self.llm.ask(prompt)

        data = json.loads(response)

        return SummaryEntry(
            summary=data.get("summary", ""),
            lessons=data.get("lessons", []),
            confidence=float(data.get("confidence", 0.0)),
            importance=data.get("importance", "low"),
        )
