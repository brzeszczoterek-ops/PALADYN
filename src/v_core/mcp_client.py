from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    stdio_client,
)


class MCPClient:

    def __init__(
        self,
        server_command: list[str],
    ):

        self.server_command = server_command

    @asynccontextmanager
    async def session(self):

        params = StdioServerParameters(
            command=self.server_command[0],
            args=self.server_command[1:],
        )

        async with stdio_client(params) as (
            read_stream,
            write_stream,
        ):

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                print("[MCP] Initializing session...")

                await asyncio.wait_for(
                    session.initialize(),
                    timeout=15,
                )

                print("[MCP] Session ready")

                yield session

    async def list_tools(self):

        async with self.session() as session:

            print("[MCP] Requesting tools...")

            result = await asyncio.wait_for(
                session.list_tools(),
                timeout=15,
            )

            print("[MCP] Tools received")

        return result

    async def tool_info(
        self,
        name: str,
    ):

        result = await self.list_tools()

        for tool in result.tools:

            if tool.name == name:
                return tool

        return None
