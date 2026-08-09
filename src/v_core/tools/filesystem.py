from __future__ import annotations

from ..mcp_client import MCPClient


class Filesystem:

    def __init__(self, client: MCPClient):
        self.client = client

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

        if not result.content:
            return "OK"

        text = []

        for item in result.content:
            if hasattr(item, "text"):
                text.append(item.text)

        return "\n".join(text)
