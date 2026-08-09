from __future__ import annotations

from .config import Config
from .mcp_client import MCPClient


class MCPRegistry:

    def __init__(
        self,
        config: Config,
    ):

        self.clients: dict[str, MCPClient] = {
            "filesystem": MCPClient(
                config.filesystem_server
            ),
            "browser": MCPClient(
                config.browser_server
            ),
        }

    async def list_tools(self) -> dict[str, list[str]]:

        result: dict[str, list[str]] = {}

        for name, client in self.clients.items():

            tools = await client.list_tools()

            result[name] = [
                tool.name
                for tool in tools.tools
            ]

        return result

    async def tool_info(
        self,
        server: str,
        tool: str,
    ):

        client = self.clients[server]

        return await client.tool_info(tool)

    async def call(
        self,
        server: str,
        tool: str,
        arguments: dict,
    ):

        client = self.clients[server]

        async with client.session() as session:

            result = await session.call_tool(
                tool,
                arguments,
            )

        if not result.content:
            return ""

        text: list[str] = []

        for item in result.content:

            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)
