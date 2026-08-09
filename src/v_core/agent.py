from __future__ import annotations

import asyncio

from .config import Config
from .llm import LLM
from .mcp_tools import MCPTools
from .tool_dispatcher import ToolDispatcher
from .memory.memory_engine import MemoryEngine
from .capability_dispatcher import CapabilityDispatcher
from .capabilities.research import ResearchTask


class Agent:

    def __init__(
        self,
        config: Config,
        memory: MemoryEngine,
    ):

        self.config = config

        self.llm = LLM()

        self.tools = MCPTools(config)

        self.dispatcher = ToolDispatcher(
            llm=self.llm,
            tools=self.tools,
        )

        self.memory = memory

        self.capabilities = CapabilityDispatcher()

        self.research = ResearchTask(
            self.dispatcher,
        )

    async def run(
        self,
        prompt: str,
    ) -> str:

        capability = self.capabilities.dispatch(prompt)

        #
        # Tool path
        #

        if capability == CapabilityDispatcher.TOOLS:

            tool_result = await self.dispatcher.dispatch(
                prompt
            )

            if tool_result is not None:

                tool_result = str(tool_result)

                self.memory.session.add(
                    "task",
                    {
                        "task": prompt,
                        "result": tool_result,
                    },
                )

                asyncio.create_task(
                    self.memory.process(
                        prompt,
                        tool_result,
                    )
                )

                return tool_result

        #
        # Research path
        #

        if capability == CapabilityDispatcher.RESEARCH:

            answer = await self.research.run(
                prompt,
            )

            self.memory.session.add(
                "task",
                {
                    "task": prompt,
                    "result": answer,
                },
            )

            asyncio.create_task(
                self.memory.process(
                    prompt,
                    answer,
                )
            )

            return answer

        #
        # Chat path
        #

        messages = [
            {
                "role": "system",
                "content": self.llm.config.system_prompt,
            }
        ]

        messages.extend(
            self.memory.session.messages(limit=10)
        )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        answer = await self.llm.ask(
            messages=messages,
        )

        self.memory.session.add(
            "task",
            {
                "task": prompt,
                "result": answer,
            },
        )

        asyncio.create_task(
            self.memory.process(
                prompt,
                answer,
            )
        )

        return answer
