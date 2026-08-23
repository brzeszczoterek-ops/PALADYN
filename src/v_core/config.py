from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Config:
    workspace: Path

    filesystem_server: list[str]

    browser_server: list[str]


def load_config() -> Config:

    load_dotenv()

    project_root = Path.cwd()

    workspace_value = os.getenv("V_CORE_MCP_FILESYSTEM", "agent_workspace")
    workspace = Path(workspace_value).expanduser()
    if not workspace.is_absolute():
        workspace = project_root / workspace

    workspace.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Config(
        workspace=workspace,

        filesystem_server=[
            "npx",
            "@modelcontextprotocol/server-filesystem",
            str(workspace),
        ],

        browser_server=[
            "npx",
            "@playwright/mcp",
            "--browser=firefox",
        ],
    )
