from __future__ import annotations

import re

from ..tool_dispatcher import ToolDispatcher
from ..mcp_tools import MCPTools
from ..llm import LLM


class ResearchTask:

    def __init__(
        self,
        dispatcher: ToolDispatcher,
    ):

        self.dispatcher = dispatcher

    def _extract_url(
        self,
        prompt: str,
    ) -> str | None:

        match = re.search(
            r"https?://\S+",
            prompt,
        )

        if match is None:
            return None

        return match.group(0).rstrip(").,;]\"'")

    async def run(
        self,
        prompt: str,
    ) -> str:

        tools = self.dispatcher.tools
        llm = self.dispatcher.llm

        url = self._extract_url(prompt)

        if url is None:
            return (
                "Nie znalazłem adresu URL w poleceniu. "
                "Podaj pełny link do strony."
            )

        await tools.browser_call(
            "browser_navigate",
            {
                "url": url,
            },
        )

        snapshot = await tools.browser_call(
            "browser_snapshot",
            {},
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. "
                    "Answer only using the provided browser snapshot. "
                    "Be concise, factual, and helpful."
                ),
            },
            {
                "role": "user",
                "content": f"""User request:

{prompt}

Browser snapshot:

{snapshot}
""",
            },
        ]

        answer = await llm.ask(
            messages=messages,
        )

        return answer or "Nie udało się wykonać researchu."
