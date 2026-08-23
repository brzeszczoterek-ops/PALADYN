from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Config:
    workspace: Path

    memory_root: Path

    autonomy_root: Path

    evm_profile: str

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

    memory_root_value = os.getenv("PALADYN_MEMORY_ROOT", "memory")
    memory_root = Path(memory_root_value).expanduser()
    if not memory_root.is_absolute():
        memory_root = project_root / memory_root
    memory_root.mkdir(parents=True, exist_ok=True)

    autonomy_root_value = os.getenv("PALADYN_AUTONOMY_ROOT", "autonomy")
    autonomy_root = Path(autonomy_root_value).expanduser()
    if not autonomy_root.is_absolute():
        autonomy_root = project_root / autonomy_root
    autonomy_root.mkdir(parents=True, exist_ok=True)

    evm_profile = os.getenv("PALADYN_EVM_PROFILE", "owner_lab").strip().lower()
    if evm_profile not in {"client", "owner_lab"}:
        raise ValueError("PALADYN_EVM_PROFILE must be 'client' or 'owner_lab'")

    return Config(
        workspace=workspace,

        memory_root=memory_root,

        autonomy_root=autonomy_root,

        evm_profile=evm_profile,

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
