from __future__ import annotations

from ..llm import LLM
from ..utils import parse_llm_json
from .models import (
    ExperienceEntry,
    MemoryKind,
    MemorySource,
    ReflectionEntry,
    KnowledgeEntry,
)


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

You are converting a reflection into a reusable experience.

Current reflection:

Summary:
{reflection.summary}

Lesson:
{reflection.lesson}

Importance:
{reflection.importance}

Remember:
{reflection.remember}

Memory kind:
{reflection.kind.value}

Memory source:
{reflection.source.value}

Previous experiences:

{previous}

Current knowledge:

{knowledge}

Your task is to determine whether this reflection represents
a meaningful reusable experience.

Rules:

- Do not create an experience from information that is purely temporary
  or irrelevant to future behaviour.
- Preserve the original meaning of the reflection.
- Do not upgrade uncertainty into certainty.
- Do not change a directly observed or directly told fact into a verified fact.
- Do not invent information that is not present in the reflection,
  previous experiences, or knowledge.
- If the reflection does not contain useful reusable information,
  return an empty summary and lesson.
- Confidence must reflect the reliability of the available information.
- Importance must reflect the practical value of remembering it.
- Keep the original memory kind unless the available context clearly
  justifies a more appropriate classification.
- Keep the original source unless stronger evidence in the provided context
  justifies changing it.
- A self-generated inference must remain distinguishable from an observed
  or directly stated fact.

Return ONLY valid JSON.

Schema:

{{
    "summary": "...",
    "lesson": "...",
    "confidence": 0.0,
    "importance": "low | medium | high",
    "kind": "fact | preference | lesson | experience | relationship_event",
    "source": "observed | directly_told | inferred | self_generated | verified"
}}

Output ONLY JSON.
""".strip()

        response = await self.llm.ask(prompt, max_tokens=192)

        data = parse_llm_json(
            response,
            default={
                "summary": "",
                "lesson": "",
                "confidence": 0.0,
                "importance": "low",
                "kind": reflection.kind.value,
                "source": reflection.source.value,
            },
        )

        try:
            kind = MemoryKind(
                data.get(
                    "kind",
                    reflection.kind.value,
                )
            )
        except ValueError:
            kind = reflection.kind

        try:
            source = MemorySource(
                data.get(
                    "source",
                    reflection.source.value,
                )
            )
        except ValueError:
            source = reflection.source

        confidence = float(
            data.get(
                "confidence",
                0.0,
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        return ExperienceEntry(
            summary=data.get("summary", ""),
            lesson=data.get("lesson", ""),
            confidence=confidence,
            importance=data.get(
                "importance",
                reflection.importance,
            ),
            kind=kind,
            source=source,
        )
