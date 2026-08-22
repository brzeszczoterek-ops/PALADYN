from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Config:
    workspace: Path

    filesystem_server: list[str]

    browser_server: list[str]


def load_config() -> Config:

    project_root = Path.cwd()

    workspace = (
        project_root / "agent_workspace"
    )

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
