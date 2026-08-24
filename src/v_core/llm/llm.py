from __future__ import annotations

import os
from collections.abc import AsyncIterator
import re

from openai import AsyncOpenAI

from .llm_config import load_llm_config


_WORD = re.compile(r"[\w']+", re.UNICODE)


def repetition_start(text: str) -> int | None:
    """Return the character offset where a clear generation loop begins."""

    matches = list(_WORD.finditer(text.casefold()))
    words = [match.group(0) for match in matches]
    if len(words) < 6:
        return None

    # Two long repeated spans, three shorter phrases, or six repeated tokens are
    # strong enough evidence to stop. Longer patterns are checked first so the
    # first complete repeated block is retained rather than a tiny suffix.
    starts: list[int] = []
    for repetitions, minimum_block in ((2, 8), (3, 3), (6, 1)):
        maximum_block = min(96, len(words) // repetitions)
        for block_size in range(maximum_block, minimum_block - 1, -1):
            tail = words[-block_size:]
            if all(
                words[
                    -block_size * offset : -block_size * (offset - 1)
                    if offset > 1
                    else None
                ]
                == tail
                for offset in range(2, repetitions + 1)
            ):
                second_block = len(words) - block_size * (repetitions - 1)
                starts.append(matches[second_block].start())
                break
    return min(starts) if starts else None


def trim_repetition(text: str) -> str:
    start = repetition_start(text)
    if start is None:
        return text
    return text[:start].rstrip()


class LLM:

    def __init__(self, api_key: str | None = None):

        self.config = load_llm_config()

        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=api_key or os.getenv("V_CORE_API_KEY", "local"),
            timeout=float(os.getenv("V_CORE_TIMEOUT", "120")),
        )

    async def ask(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        max_tokens: int | None = None,
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

        messages = self._normalize_system_messages(messages)

        response = await self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            messages=messages,
            max_tokens=max_tokens or int(os.getenv("V_CORE_MAX_TOKENS", "512")),
        )

        message = response.choices[0].message

        if getattr(message, "content", None):
            return trim_repetition(message.content)

        return ""

    async def stream(
        self,
        *,
        messages: list,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        messages = self._normalize_system_messages(messages)
        response = await self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            messages=messages,
            max_tokens=max_tokens or int(os.getenv("V_CORE_MAX_TOKENS", "512")),
            stream=True,
        )

        generated = ""
        async for chunk in response:
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                candidate = generated + content
                loop_start = repetition_start(candidate)
                if loop_start is not None:
                    if loop_start > len(generated):
                        safe_tail = candidate[len(generated) : loop_start].rstrip()
                        if safe_tail:
                            yield safe_tail
                    break
                generated = candidate
                yield content

    @staticmethod
    def _normalize_system_messages(messages: list) -> list:
        """Keep exactly one system message at the beginning for GGUF templates."""

        system_parts: list[str] = []
        conversation: list[dict] = []
        for message in messages:
            copied = dict(message)
            if copied.get("role") == "system":
                content = str(copied.get("content", "")).strip()
                if content:
                    system_parts.append(content)
            else:
                if (
                    conversation
                    and copied.get("role") in {"user", "assistant"}
                    and conversation[-1].get("role") == copied.get("role")
                ):
                    previous = str(conversation[-1].get("content", ""))
                    current = str(copied.get("content", ""))
                    conversation[-1]["content"] = "\n\n".join(
                        part for part in (previous, current) if part
                    )
                else:
                    conversation.append(copied)

        if not system_parts:
            return conversation
        return [
            {
                "role": "system",
                "content": "\n\n".join(system_parts),
            },
            *conversation,
        ]
