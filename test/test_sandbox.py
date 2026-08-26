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
async def test_seccomp_fallback_runs_code_but_denies_socket_creation(
    tmp_path: Path,
) -> None:
    backend = BubblewrapBackend(force_seccomp_network_filter=True)
    ordinary = await backend.run(
        SandboxSpec(
            command=("/usr/bin/python3", "-c", "print(6 * 7)"),
            workspace=tmp_path / "ordinary",
        )
    )
    network = await backend.run(
        SandboxSpec(
            command=(
                "/usr/bin/python3",
                "-c",
                "import socket; socket.socket()",
            ),
            workspace=tmp_path / "network",
        )
    )

    assert ordinary.succeeded
    assert ordinary.stdout.strip() == "42"
    assert ordinary.backend == "bubblewrap+seccomp-netblock"
    assert not network.succeeded
    assert "PermissionError" in network.stderr


def test_share_network_cannot_be_enabled_without_seccomp(tmp_path: Path) -> None:
    backend = BubblewrapBackend()
    spec = SandboxSpec(command=("/usr/bin/true",), workspace=tmp_path)

    with pytest.raises(SandboxPolicyError, match="requires a seccomp filter"):
        backend._build_argv(spec, tmp_path.resolve(), share_network=True)


@pytest.mark.asyncio
async def test_loopback_permission_error_retries_with_seccomp_fallback(
    tmp_path: Path,
) -> None:
    wrapper = tmp_path / "bwrap-once"
    marker = tmp_path / "failed-once"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f"if [[ ! -e {marker!s} ]]; then\n"
        f"  touch {marker!s}\n"
        "  echo 'bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted' >&2\n"
        "  exit 1\n"
        "fi\n"
        "exec /usr/bin/bwrap \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    result = await BubblewrapBackend(executable=wrapper).run(
        SandboxSpec(
            command=("/usr/bin/python3", "-c", "print('recovered')"),
            workspace=tmp_path / "workspace",
        )
    )

    assert result.succeeded
    assert result.stdout.strip() == "recovered"
    assert result.backend == "bubblewrap+seccomp-netblock"


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
async def test_bubblewrap_stops_total_workspace_growth(tmp_path: Path) -> None:
    result = await BubblewrapBackend().run(
        SandboxSpec(
            command=(
                "/usr/bin/python3",
                "-c",
                (
                    "from pathlib import Path; "
                    "[Path(f'f{i}').write_bytes(b'x'*4096) for i in range(256)]"
                ),
            ),
            workspace=tmp_path / "workspace",
            limits=SandboxLimits(
                max_file_bytes=8 * 1024,
                max_workspace_bytes=32 * 1024,
            ),
        )
    )

    assert result.workspace_limited
    assert not result.succeeded


@pytest.mark.asyncio
async def test_bubblewrap_applies_process_limit(tmp_path: Path) -> None:
    result = await BubblewrapBackend().run(
        SandboxSpec(
            command=(
                "/usr/bin/python3",
                "-c",
                "import resource; print(resource.getrlimit(resource.RLIMIT_NPROC)[0])",
            ),
            workspace=tmp_path / "workspace",
            limits=SandboxLimits(max_processes=17),
        )
    )

    assert result.succeeded
    assert result.stdout.strip() == "17"


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
