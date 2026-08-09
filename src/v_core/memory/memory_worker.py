from __future__ import annotations

import json
from pathlib import Path

from ..llm import LLM
from .models import SummaryEntry
from .storage import MemoryStorage


class MemoryWorker:

    def __init__(
        self,
        llm: LLM,
        storage: MemoryStorage,
    ):

        self.llm = llm
        self.storage = storage

    async def summarize(
        self,
        limit: int = 5,
    ) -> SummaryEntry | None:

        reflections = self.storage.list("reflections")

        if len(reflections) < limit:
            return None

        selected = reflections[:limit]

        documents = [
            self.storage.load(file)
            for file in selected
        ]

        prompt = f"""
You are V.

Below are your recent reflections.

{json.dumps(documents, indent=2)}

Create ONE summary.

Return ONLY valid JSON.

Schema:

{{
    "summary": "...",
    "lessons": [
        "...",
        "...",
        "..."
    ],
    "confidence": 1.0,
    "importance": "low | medium | high"
}}

Rules:

- Merge duplicated lessons.
- Keep only reusable knowledge.
- Return ONLY JSON.
- No markdown.
""".strip()

        response = await self.llm.ask(prompt)

        data = json.loads(response)

        summary = SummaryEntry(
            summary=data.get("summary", ""),
            lessons=data.get("lessons", []),
            confidence=float(data.get("confidence", 0.0)),
            importance=data.get("importance", "low"),
        )

        filename = f"summary_{selected[-1].stem}.yaml"

        self.storage.save(
            "summaries",
            filename,
            summary,
        )

        for file in selected:
            Path(file).unlink(missing_ok=True)

        return summary
