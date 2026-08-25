from __future__ import annotations

from ..mcp_client import MCPClient


class FilesystemToolError(RuntimeError):
    """The filesystem MCP server completed the call with an error result."""


class Filesystem:

    def __init__(self, client: MCPClient):
        self.client = client

    @staticmethod
    def _raise_if_error(result: object, tool: str) -> None:
        if not bool(
            getattr(result, "isError", False)
            or getattr(result, "is_error", False)
        ):
            return
        output = "\n".join(
            str(item.text)
            for item in (getattr(result, "content", None) or [])
            if hasattr(item, "text")
        ).strip()
        raise FilesystemToolError(
            output[:2_000] or f"filesystem tool {tool} returned an error"
        )

    async def list_directory(
        self,
        path: str,
    ) -> list[str]:

        async with self.client.session() as session:

            result = await session.call_tool(
                "list_directory",
                {
                    "path": path,
                },
            )

        self._raise_if_error(result, "list_directory")

        if not result.content:
            return []

        entries = []

        for item in result.content:
            if hasattr(item, "text"):
                entries.append(item.text)

        return entries

    async def read_file(
        self,
        path: str,
    ) -> str:

        async with self.client.session() as session:

            result = await session.call_tool(
                "read_file",
                {
                    "path": path,
                },
            )

        self._raise_if_error(result, "read_file")

        if not result.content:
            return ""

        text = []

        for item in result.content:
            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)

    async def write_file(
        self,
        path: str,
        content: str,
    ) -> str:

        async with self.client.session() as session:

            result = await session.call_tool(
                "write_file",
                {
                    "path": path,
                    "content": content,
                },
            )

        self._raise_if_error(result, "write_file")

        if not result.content:
            return "OK"

        text = []

        for item in result.content:
            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)

    async def edit_file(
        self,
        path: str,
        edits: list[dict],
        dry_run: bool = False,
    ) -> str:

        async with self.client.session() as session:

            result = await session.call_tool(
                "edit_file",
                {
                    "path": path,
                    "edits": edits,
                    "dryRun": dry_run,
                },
            )

        self._raise_if_error(result, "edit_file")

        if not result.content:
            return "OK"

        text = []

        for item in result.content:
            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)

    async def move_file(
        self,
        source: str,
        destination: str,
    ) -> str:

        async with self.client.session() as session:

            result = await session.call_tool(
                "move_file",
                {
                    "source": source,
                    "destination": destination,
                },
            )

        self._raise_if_error(result, "move_file")

        if not result.content:
            return "OK"

        text = []

        for item in result.content:
            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)

    async def search_files(
        self,
        path: str,
        pattern: str,
    ) -> str:

        async with self.client.session() as session:

            result = await session.call_tool(
                "search_files",
                {
                    "path": path,
                    "pattern": pattern,
                },
            )

        self._raise_if_error(result, "search_files")

        if not result.content:
            return ""

        text = []

        for item in result.content:
            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)

    async def directory_tree(
        self,
        path: str,
    ) -> str:

        async with self.client.session() as session:

            result = await session.call_tool(
                "directory_tree",
                {
                    "path": path,
                },
            )

        self._raise_if_error(result, "directory_tree")

        if not result.content:
            return ""

        text = []

        for item in result.content:
            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)

    async def get_file_info(
        self,
        path: str,
    ) -> str:

        async with self.client.session() as session:

            result = await session.call_tool(
                "get_file_info",
                {
                    "path": path,
                },
            )

        self._raise_if_error(result, "get_file_info")

        if not result.content:
            return ""

        text = []

        for item in result.content:
            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)

    async def make_directory(
        self,
        path: str,
    ) -> str:

        async with self.client.session() as session:

            result = await session.call_tool(
                "create_directory",
                {
                    "path": path,
                },
            )

        self._raise_if_error(result, "create_directory")

        if not result.content:
            return "OK"

        text = []

        for item in result.content:
            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)
