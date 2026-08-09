from __future__ import annotations

import json

from ..llm import LLM
from .models import ExperienceEntry, ReflectionEntry, KnowledgeEntry


class Experience:

    def __init__(self, llm: LLM):
        self.llm = llm

    async def learn(
        self,
        reflection: ReflectionEntry,
        previous: list[ExperienceEntry],
        knowledge: list[KnowledgeEntry],
    ) -> ExperienceEntry:

        prompt = f"""
You are V.

Current reflection:

Summary:
{reflection.summary}

Lesson:
{reflection.lesson}

Previous experience:

{previous}

Knowledge:

{knowledge}

Your task is to determine whether this reflection
changes your experience.

Return ONLY valid JSON.

Schema:

{{
    "summary": "...",
    "lesson": "...",
    "confidence": 0.0,
    "importance": "low | medium | high"
}}

Rules:

- confidence must be between 0.0 and 1.0
- do not use markdown
- do not explain anything
- output ONLY JSON
""".strip()

        response = await self.llm.ask(prompt)

        data = json.loads(response)

        return ExperienceEntry(
            summary=data.get("summary", ""),
            lesson=data.get("lesson", ""),
            confidence=float(data.get("confidence", 0.0)),
            importance=data.get("importance", "low"),
        )
