from __future__ import annotations

from .llm import LLM
from .mcp_tools import MCPTools


ROUTER_PROMPT = """
You are a routing engine.

Your ONLY task is to decide whether the user's request
requires using an external tool.

Return ONLY:

YES

or

NO

Never explain.
Never answer the question.
""".strip()


TOOL_PROMPT = """
You are the PALADYN Tool Dispatcher.

The router has already decided that a tool IS required.

Available tools:

{tools}

Return ONLY one JSON object.

Format:

{"tool": "<tool_name>", "arguments": {}}

Filesystem examples:

{"tool": "list_directory", "arguments": {"path": "."}}
{"tool": "read_file", "arguments": {"path": "README.md"}}
{"tool": "write_file", "arguments": {"path": "test.txt", "content": "Hello"}}

Browser examples:

{"tool": "browser_navigate", "arguments": {"url": "https://openai.com"}}
{"tool": "browser_snapshot", "arguments": {}}

Rules:

- Return ONLY one JSON object.
- Never explain.
- Never answer the user.
- Never invent tool names.
- Use browser tools whenever the task involves websites, web pages, searching page content or navigation.
- Use filesystem tools only for local files.

""".strip()


class ToolDispatcher:

    def __init__(
        self,
        llm: LLM,
        tools: MCPTools,
    ):

        self.llm = llm
        self.tools = tools


    async def dispatch(
        self,
        prompt: str,
    ) -> str | None:

        #
        # Step 1
        # Does this request require tools?
        #

        decision = await self.llm.ask(
            f"""{ROUTER_PROMPT}

User request:

{prompt}
"""
        )

        decision = decision.strip().upper()

        if decision != "YES":
            return None

        #
        # Step 2
        # Which tool?
        #

        available = await self.tools.tools()

        tool_list = "\n".join(available)

        decision = await self.llm.ask(
            TOOL_PROMPT.format(
                tools=tool_list
            )
            + f"""

User request:

{prompt}
"""
        )

        decision = decision.strip()

        from .utils import parse_llm_json

        action = parse_llm_json(decision)
        tool = action.get("tool")
        arguments = action.get("arguments", {})

        if not isinstance(tool, str) or not tool.strip():
            return None
        if not isinstance(arguments, (dict, str)):
            return None

        return await self.tools.call(
            tool.strip(),
            arguments,
        )
