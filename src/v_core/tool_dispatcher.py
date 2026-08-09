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

Return ONLY one line.

Format:

TOOL:<tool_name>:<arguments>

Filesystem examples:

TOOL:ls:.
TOOL:tree:.
TOOL:cat:README.md
TOOL:mkdir:project
TOOL:move:file1.txt,file2.txt
TOOL:search:.,agent.py
TOOL:write:test.txt|Hello World

Browser examples:

TOOL:browser_navigate:https://openai.com
TOOL:browser_snapshot:
TOOL:browser_find:OpenAI
TOOL:browser_click:123
TOOL:browser_press_key:Enter

Rules:

- Return ONLY one TOOL line.
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

        if not decision.startswith("TOOL:"):
            return None

        try:

            _, tool, arguments = decision.split(
                ":",
                2,
            )

        except ValueError:

            return None

        return await self.tools.call(
            tool.strip(),
            arguments.strip(),
        )
