from __future__ import annotations

import json

from ..llm import LLM
from .models import (
    SummaryEntry,
    KnowledgeEntry,
)


class Knowledge:

    def __init__(
        self,
        llm: LLM,
    ):
        self.llm = llm

    async def update(
        self,
        summary: SummaryEntry,
        knowledge: list[KnowledgeEntry],
    ) -> KnowledgeEntry:

        prompt = f"""
You are V.

Current knowledge:

{knowledge}

New summary:

{summary}

Determine whether your long-term knowledge should change.

Return ONLY valid JSON.

Schema:

{{
    "title": "...",
    "content": "...",
    "reason": "...",
    "confidence": 0.0
}}

Rules:

- Update only if the knowledge is stable.
- Ignore one-time events.
- Confidence must be between 0.0 and 1.0.
- Output ONLY JSON.
- No markdown.
""".strip()

        response = await self.llm.ask(prompt)

        data = json.loads(response)

        return KnowledgeEntry(
            title=data.get("title", ""),
            content=data.get("content", ""),
            reason=data.get("reason", ""),
            confidence=float(data.get("confidence", 0.0)),
        )
