from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.autonomy.policy import AuthorizationDenied
from v_core.sandbox import (
    BubblewrapBackend,
    NetworkMode,
    SandboxExecutor,
    SandboxLimits,
    SandboxPolicyError,
    SandboxSpec,
)


pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None or shutil.which("prlimit") is None,
    reason="bubblewrap and prlimit are required",
)


@pytest.mark.asyncio
async def test_bubblewrap_hides_home_and_uses_private_workspace(tmp_path: Path) -> None:
    result = await BubblewrapBackend().run(
        SandboxSpec(
            command=(
                "/usr/bin/python3",
                "-c",
                "import os; print(os.getcwd()); print(os.path.exists('/home/Vi'))",
            ),
            workspace=tmp_path / "workspace",
        )
    )

    assert result.succeeded
    assert result.stdout.splitlines() == ["/workspace", "False"]


@pytest.mark.asyncio
async def test_bubblewrap_has_no_host_network(tmp_path: Path) -> None:
    result = await BubblewrapBackend().run(
        SandboxSpec(
            command=(
                "/usr/bin/python3",
                "-c",
                "import socket; s=socket.socket(); print(s.connect_ex(('1.1.1.1', 53)))",
            ),
            workspace=tmp_path / "workspace",
        )
    )

    assert result.succeeded
    assert result.stdout.strip() != "0"


@pytest.mark.asyncio
async def test_bubblewrap_enforces_wall_clock_timeout(tmp_path: Path) -> None:
    result = await BubblewrapBackend().run(
        SandboxSpec(
            command=("/usr/bin/python3", "-c", "import time; time.sleep(10)"),
            workspace=tmp_path / "workspace",
            limits=SandboxLimits(timeout_seconds=0.1),
        )
    )

    assert result.timed_out
    assert not result.succeeded


@pytest.mark.asyncio
async def test_bubblewrap_stops_unbounded_output(tmp_path: Path) -> None:
    result = await BubblewrapBackend().run(
        SandboxSpec(
            command=("/usr/bin/python3", "-c", "print('x' * 100000)"),
            workspace=tmp_path / "workspace",
            limits=SandboxLimits(max_output_bytes=1_024),
        )
    )

    assert result.output_limited
    assert not result.succeeded


@pytest.mark.asyncio
async def test_backend_fails_closed_for_unimplemented_network_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(SandboxPolicyError):
        await BubblewrapBackend().run(
            SandboxSpec(
                command=("/usr/bin/true",),
                workspace=tmp_path / "workspace",
                network=NetworkMode.ALLOWLIST_PROXY,
            )
        )


@pytest.mark.asyncio
async def test_executor_confines_workspace_to_authorization_envelope(
    tmp_path: Path,
) -> None:
    task_workspace = tmp_path / "task"
    envelope = AuthorizationEnvelope(workspace=str(task_workspace))
    executor = SandboxExecutor(
        AuthorizationGuard(tmp_path, envelope),
        BubblewrapBackend(),
    )

    with pytest.raises(AuthorizationDenied):
        await executor.execute(
            SandboxSpec(
                command=("/usr/bin/true",),
                workspace=tmp_path / "outside",
            )
        )
