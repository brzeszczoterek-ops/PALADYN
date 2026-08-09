from __future__ import annotations

from openai import AsyncOpenAI

from .llm_config import CURRENT


class LLM:

    def __init__(self, api_key: str = "local"):

        self.config = CURRENT

        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=api_key,
        )

    async def ask(
        self,
        prompt: str | None = None,
        messages: list | None = None,
    ) -> str:

        if messages is None:

            messages = [
                {
                    "role": "system",
                    "content": self.config.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt or "",
                },
            ]

        response = await self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            messages=messages,
        )

        message = response.choices[0].message

        if getattr(message, "content", None):
            return message.content

        return ""
