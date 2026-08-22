from __future__ import annotations

from ..llm import LLM
from ..utils import parse_llm_json
from .models import MemoryKind, MemorySource, ReflectionEntry


class Reflection:

    def __init__(self, llm: LLM):
        self.llm = llm

    async def reflect(
        self,
        task: str,
        result: str,
    ) -> ReflectionEntry:

        prompt = f"""
You are V.

You have just completed a task.

Task:

{task}

Result:

{result}

You are performing a critical self-review.

Your purpose is to improve future behaviour and identify information
that may be worth retaining.

Rules:

- Never assume that the task was completed correctly.
- Base your reflection ONLY on the provided task and result.
- If the result appears incorrect, incomplete or hallucinated, explicitly say so.
- If a tool should have been used instead of answering from memory, mention it.
- If the correctness cannot be verified, state that clearly.
- Extract ONE practical lesson that will improve future behaviour.
- Set remember=true ONLY if the lesson or observation is likely to be useful
  in future tasks.
- Use "high" importance only for information that significantly improves
  future behaviour.
- Classify the most important reusable information using exactly one memory kind.
- Do not treat an inference as a directly observed fact.
- Do not claim something was verified unless the provided result supports that.
- If the information comes from your own reasoning rather than directly
  from the task or result, use source "self_generated".
- Prefer "lesson" when the reusable information concerns how V should behave
  or perform tasks.
- Use "fact" for a stable factual observation.
- Use "preference" for a stable preference expressed or demonstrated by Boss.
- Use "experience" for a meaningful event that should be remembered as an event.
- Use "relationship_event" only when the information meaningfully concerns
  the relationship between V and Boss.

Return ONLY valid JSON.

Schema:

{{
    "summary": "...",
    "lesson": "...",
    "importance": "low | medium | high",
    "remember": true,
    "kind": "fact | preference | lesson | experience | relationship_event",
    "source": "observed | directly_told | inferred | self_generated | verified"
}}

Rules for classification:

- "directly_told" means the information was explicitly stated in the task.
- "observed" means it can be directly established from the provided result.
- "verified" means the result explicitly contains evidence that verifies it.
- "inferred" means V derived it from the available information.
- "self_generated" means it primarily comes from V's own reasoning.
- Never use "verified" merely because the answer sounds convincing.
- When uncertain, prefer a weaker source classification.

Output ONLY JSON.
""".strip()

        response = await self.llm.ask(prompt)

        data = parse_llm_json(
            response,
            default={
                "summary": result[:200],
                "lesson": "",
                "importance": "low",
                "remember": False,
                "kind": MemoryKind.LESSON.value,
                "source": MemorySource.SELF_GENERATED.value,
            },
        )

        try:
            kind = MemoryKind(
                data.get(
                    "kind",
                    MemoryKind.LESSON.value,
                )
            )
        except ValueError:
            kind = MemoryKind.LESSON

        try:
            source = MemorySource(
                data.get(
                    "source",
                    MemorySource.SELF_GENERATED.value,
                )
            )
        except ValueError:
            source = MemorySource.SELF_GENERATED

        return ReflectionEntry(
            task=task,
            result=result,
            summary=data.get("summary", ""),
            lesson=data.get("lesson", ""),
            importance=data.get("importance", "low"),
            remember=bool(data.get("remember", False)),
            kind=kind,
            source=source,
        )
