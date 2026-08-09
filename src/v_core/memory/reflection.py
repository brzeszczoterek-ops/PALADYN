from __future__ import annotations

from ..llm import LLM
from ..utils import parse_llm_json
from .models import ReflectionEntry


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

Your purpose is to improve future behaviour.

Rules:

- Never assume that the task was completed correctly.
- Base your reflection ONLY on the provided task and result.
- If the result appears incorrect, incomplete or hallucinated, explicitly say so.
- If a tool should have been used instead of answering from memory, mention it.
- If the correctness cannot be verified, state that clearly.
- Extract ONE practical lesson that will improve future behaviour.
- Set remember=true ONLY if the lesson is likely to be useful in future tasks.
- Use "high" importance only for lessons that significantly improve the agent.

Return ONLY valid JSON.

Schema:

{{
    "summary": "...",
    "lesson": "...",
    "importance": "low | medium | high",
    "remember": true
}}
""".strip()

        response = await self.llm.ask(prompt)

        data = parse_llm_json(
            response,
            default={
                "summary": result[:200],
                "lesson": "",
                "importance": "low",
                "remember": False,
            },
        )

        return ReflectionEntry(
            task=task,
            result=result,
            summary=data.get("summary", ""),
            lesson=data.get("lesson", ""),
            importance=data.get("importance", "low"),
            remember=data.get("remember", False),
        )
