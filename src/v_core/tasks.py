from __future__ import annotations

import json

from .llm import LLM
from .mcp_tools import MCPTools


class WriteFileTask:

    def __init__(
        self,
        llm: LLM,
        tools: MCPTools,
    ):
        self.llm = llm
        self.tools = tools

    async def run(
        self,
        path: str,
        instruction: str,
    ) -> str:

        prompt = f"""
You are an expert software engineer.

Generate the COMPLETE contents of a single source file.

Return ONLY source code.

Task:

{instruction}
""".strip()

        content = await self.llm.ask(prompt)

        return await self.tools.write(
            path,
            content,
        )


class EditFileTask:

    def __init__(
        self,
        llm: LLM,
        tools: MCPTools,
    ):
        self.llm = llm
        self.tools = tools

    async def run(
        self,
        path: str,
        instruction: str,
    ) -> str:

        original = await self.tools.cat(path)

        prompt = f"""
You are editing an existing source file.

Current file:

----------------
{original}
----------------

Task:

{instruction}

Return ONLY valid JSON.

Format:

{{
    "oldText": "...",
    "newText": "..."
}}

Rules:

- oldText MUST match the current file exactly.
- newText is the replacement.
- Return ONE edit only.
- No markdown.
- No explanations.
- Output ONLY JSON.
""".strip()

        response = await self.llm.ask(prompt)

        try:
            patch = json.loads(response)
        except Exception:
            return f"Invalid JSON returned by LLM:\n\n{response}"

        return await self.tools.edit(
            path,
            [
                {
                    "oldText": patch["oldText"],
                    "newText": patch["newText"],
                }
            ],
        )


class MoveFileTask:

    def __init__(
        self,
        tools: MCPTools,
    ):
        self.tools = tools

    async def run(
        self,
        source: str,
        destination: str,
    ) -> str:

        return await self.tools.move(
            source,
            destination,
        )


class SearchFilesTask:

    def __init__(
        self,
        tools: MCPTools,
    ):
        self.tools = tools

    async def run(
        self,
        path: str,
        pattern: str,
    ) -> str:

        return await self.tools.search(
            path,
            pattern,
        )


class DirectoryTreeTask:

    def __init__(
        self,
        tools: MCPTools,
    ):
        self.tools = tools

    async def run(
        self,
        path: str,
    ) -> str:

        return await self.tools.tree(path)


class FileInfoTask:

    def __init__(
        self,
        tools: MCPTools,
    ):
        self.tools = tools

    async def run(
        self,
        path: str,
    ) -> str:

        return await self.tools.info(path)
