from __future__ import annotations

from .config import Config
from .mcp_client import MCPClient
from .tools.filesystem import Filesystem


class MCPTools:

    def __init__(
        self,
        config: Config,
    ):

        #
        # MCP clients
        #

        self.filesystem_client = MCPClient(
            config.filesystem_server
        )

        self.browser_client = MCPClient(
            config.browser_server
        )

        #
        # Browser session state
        #

        self.browser_session = None
        self.browser_ready = False

        #
        # Wrappers
        #

        self.filesystem = Filesystem(
            self.filesystem_client
        )

    async def ensure_browser_session(self) -> None:

        if self.browser_ready and self.browser_session is not None:
            return

        params = self.browser_client.server_command

        from mcp import ClientSession
        from mcp.client.stdio import (
            StdioServerParameters,
            stdio_client,
        )

        self._browser_stdio = stdio_client(
            StdioServerParameters(
                command=params[0],
                args=params[1:],
            )
        )

        self._browser_streams = await self._browser_stdio.__aenter__()
        read_stream, write_stream = self._browser_streams

        self.browser_session = ClientSession(
            read_stream,
            write_stream,
        )

        await self.browser_session.__aenter__()

        print("[MCP] Initializing browser session...")

        await self.browser_session.initialize()

        print("[MCP] Browser session ready")

        self.browser_ready = True

    async def close_browser_session(self) -> None:

        if not self.browser_ready or self.browser_session is None:
            return

        try:
            await self.browser_session.__aexit__(None, None, None)
        finally:
            if hasattr(self, "_browser_stdio"):
                await self._browser_stdio.__aexit__(None, None, None)

            self.browser_session = None
            self.browser_ready = False

    #
    # Filesystem shortcuts
    #

    async def ls(
        self,
        path: str = ".",
    ) -> list[str]:

        return await self.filesystem.list_directory(
            path
        )

    async def cat(
        self,
        path: str,
    ) -> str:

        return await self.filesystem.read_file(
            path
        )

    async def write(
        self,
        path: str,
        content: str,
    ) -> str:

        return await self.filesystem.write_file(
            path,
            content,
        )

    async def edit(
        self,
        path: str,
        edits: list[dict],
        dry_run: bool = False,
    ) -> str:

        return await self.filesystem.edit_file(
            path,
            edits,
            dry_run,
        )

    async def mkdir(
        self,
        path: str,
    ) -> str:

        return await self.filesystem.make_directory(
            path
        )

    async def move(
        self,
        source: str,
        destination: str,
    ) -> str:

        return await self.filesystem.move_file(
            source,
            destination,
        )

    async def search(
        self,
        path: str,
        pattern: str,
    ) -> str:

        return await self.filesystem.search_files(
            path,
            pattern,
        )

    async def tree(
        self,
        path: str = ".",
    ) -> str:

        return await self.filesystem.directory_tree(
            path
        )

    async def info(
        self,
        path: str,
    ) -> str:

        return await self.filesystem.get_file_info(
            path
        )

    #
    # Browser MCP
    #

    async def browser_call(
        self,
        tool: str,
        arguments: dict,
    ) -> str:

        await self.ensure_browser_session()

        result = await self.browser_session.call_tool(
            tool,
            arguments,
        )

        if not result.content:
            return ""

        text = []

        for item in result.content:
            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)

    #
    # Metadata
    #

    async def tools(self):

        filesystem_tools = await self.filesystem_client.list_tools()
        browser_tools = await self.browser_client.list_tools()

        return [
            tool.name for tool in filesystem_tools.tools
        ] + [
            tool.name for tool in browser_tools.tools
        ]

    async def tool_info(
        self,
        name: str,
    ):
        filesystem_tools = await self.filesystem_client.list_tools()

        for tool in filesystem_tools.tools:
            if tool.name == name:
                return tool

        browser_tools = await self.browser_client.list_tools()

        for tool in browser_tools.tools:
            if tool.name == name:
                return tool

        return None

    #
    # Dispatcher
    #

    async def call(
        self,
        tool: str,
        arguments: str = "",
    ):

        match tool:

            #
            # Filesystem
            #

            case "ls":
                return await self.ls(arguments or ".")

            case "tree":
                return await self.tree(arguments or ".")

            case "cat":
                return await self.cat(arguments)

            case "mkdir":
                return await self.mkdir(arguments)

            case "info":
                return await self.info(arguments)

            case "search":

                path, pattern = arguments.split(",", 1)

                return await self.search(
                    path.strip(),
                    pattern.strip(),
                )

            case "write":

                path, content = arguments.split("|", 1)

                return await self.write(
                    path.strip(),
                    content,
                )

            case "move":

                source, destination = arguments.split(",", 1)

                return await self.move(
                    source.strip(),
                    destination.strip(),
                )

            #
            # Browser
            #

            case "browser_navigate":
                return await self.browser_call(
                    tool,
                    {
                        "url": arguments,
                    },
                )

            case "browser_snapshot":
                return await self.browser_call(
                    tool,
                    {},
                )

            case "browser_find":
                return await self.browser_call(
                    tool,
                    {
                        "text": arguments,
                    },
                )

            case _:
                raise ValueError(
                    f"Unknown MCP tool: {tool}"
                )
