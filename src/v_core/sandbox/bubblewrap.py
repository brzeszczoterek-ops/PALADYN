from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
import shutil
import signal
import time

from .models import NetworkMode, SandboxResult, SandboxSpec


class SandboxUnavailable(RuntimeError):
    pass


class SandboxPolicyError(PermissionError):
    pass


class _OutputLimitExceeded(RuntimeError):
    pass


class _WorkspaceLimitExceeded(RuntimeError):
    pass


_SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BLOCKED_ENV = {
    "BASH_ENV",
    "ENV",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
    "SSH_AUTH_SOCK",
}


class BubblewrapBackend:
    """Fail-closed, offline execution using the external bubblewrap binary.

    The guest gets a minimal read-only operating-system view, a private PID and
    network namespace, an ephemeral /tmp, and one writable task workspace. It
    never inherits the host environment, home directory, wallet files, SSH
    agent, Docker socket, or PALADYN memory.
    """

    name = "bubblewrap"

    def __init__(
        self,
        executable: str | Path | None = None,
        resource_limiter: str | Path | None = None,
    ) -> None:
        discovered = str(executable) if executable else shutil.which("bwrap")
        if not discovered:
            raise SandboxUnavailable("bubblewrap (bwrap) is not installed")
        self.executable = Path(discovered).resolve()

        limiter = (
            str(resource_limiter)
            if resource_limiter
            else shutil.which("prlimit")
        )
        if not limiter:
            raise SandboxUnavailable("prlimit is not installed")
        self.resource_limiter = Path(limiter).resolve()

    async def run(self, spec: SandboxSpec) -> SandboxResult:
        workspace = Path(spec.workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        self._validate_spec(spec, workspace)
        argv = self._build_argv(spec, workspace)

        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        timed_out = False
        output_limited = False
        workspace_limited = False
        stdout = b""
        stderr = b""
        try:
            try:
                stdout, stderr = await asyncio.wait_for(
                    self._collect_output(
                        process,
                        spec.limits.max_output_bytes,
                        workspace,
                        spec.limits.max_workspace_bytes,
                    ),
                    timeout=spec.limits.timeout_seconds,
                )
            except TimeoutError:
                timed_out = True
                await self._kill(process)
                stdout, stderr = await process.communicate()
            except _OutputLimitExceeded:
                output_limited = True
                await self._kill(process)
                stdout, stderr = await process.communicate()
            except _WorkspaceLimitExceeded:
                workspace_limited = True
                await self._kill(process)
                stdout, stderr = await process.communicate()
        except asyncio.CancelledError:
            await self._kill(process)
            raise

        limit = spec.limits.max_output_bytes
        return SandboxResult(
            command=spec.command,
            exit_code=process.returncode,
            stdout=self._decode(stdout[:limit]),
            stderr=self._decode(stderr[:limit]),
            duration_seconds=time.monotonic() - started,
            backend=self.name,
            timed_out=timed_out,
            output_limited=output_limited,
            workspace_limited=workspace_limited,
        )

    def _validate_spec(self, spec: SandboxSpec, workspace: Path) -> None:
        if spec.network is not NetworkMode.OFFLINE:
            raise SandboxPolicyError(
                f"network mode {spec.network.value!r} has no enforcing backend"
            )

        workdir = (workspace / spec.working_directory).resolve()
        if not workdir.is_relative_to(workspace):
            raise SandboxPolicyError("working directory escapes the workspace")
        workdir.mkdir(parents=True, exist_ok=True)
        if self._workspace_size(workspace) > spec.limits.max_workspace_bytes:
            raise SandboxPolicyError("workspace exceeds configured size limit")

        for name, value in spec.environment.items():
            if not _SAFE_ENV_NAME.fullmatch(name) or name in _BLOCKED_ENV:
                raise SandboxPolicyError(f"unsafe environment variable: {name}")
            if "\x00" in value:
                raise SandboxPolicyError(f"environment variable contains NUL: {name}")

        for source in spec.read_only_inputs:
            resolved = Path(source).resolve(strict=True)
            if resolved == workspace or resolved.is_relative_to(workspace):
                raise SandboxPolicyError(
                    "workspace content is already mounted; do not duplicate it as input"
                )

    def _build_argv(self, spec: SandboxSpec, workspace: Path) -> list[str]:
        limits = spec.limits
        argv = [
            str(self.resource_limiter),
            f"--as={limits.memory_mb * 1024 * 1024}",
            f"--cpu={limits.cpu_seconds}",
            f"--fsize={limits.max_file_bytes}",
            f"--nofile={limits.max_open_files}",
            "--",
            str(self.executable),
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/sbin",
            "/sbin",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/home",
            "--dir",
            "/nonexistent",
            "--bind",
            str(workspace),
            "/workspace",
        ]

        for index, source in enumerate(spec.read_only_inputs):
            resolved = Path(source).resolve(strict=True)
            guest = f"/inputs/{index}-{resolved.name or 'input'}"
            argv.extend(("--ro-bind", str(resolved), guest))

        guest_workdir = Path("/workspace", spec.working_directory).as_posix()
        argv.extend(("--chdir", guest_workdir))
        base_environment = {
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "TMPDIR": "/tmp",
        }
        for name, value in {**base_environment, **spec.environment}.items():
            argv.extend(("--setenv", name, value))

        argv.append("--")
        # RLIMIT_NPROC is accounted against the real user in the caller's user
        # namespace. Applying it to bwrap itself makes sandbox startup depend on
        # how many processes the desktop user already owns; a busy workstation
        # can therefore prevent bwrap from creating its private namespaces at
        # all. Apply the process limit only after bwrap has entered the isolated
        # user/PID namespaces so it constrains the guest workload, not PALADYN's
        # host session.
        argv.extend(
            (
                "/usr/bin/prlimit",
                f"--nproc={limits.max_processes}",
                "--",
                *spec.command,
            )
        )
        return argv

    async def _collect_output(
        self,
        process: asyncio.subprocess.Process,
        maximum: int,
        workspace: Path,
        maximum_workspace: int,
    ) -> tuple[bytes, bytes]:
        assert process.stdout is not None
        assert process.stderr is not None
        total = 0
        lock = asyncio.Lock()

        async def read(stream: asyncio.StreamReader) -> bytes:
            nonlocal total
            chunks: list[bytes] = []
            while chunk := await stream.read(64 * 1024):
                async with lock:
                    total += len(chunk)
                    if total > maximum:
                        raise _OutputLimitExceeded
                chunks.append(chunk)
            return b"".join(chunks)

        stdout_task = asyncio.create_task(read(process.stdout))
        stderr_task = asyncio.create_task(read(process.stderr))
        wait_task = asyncio.create_task(process.wait())

        async def monitor_workspace() -> None:
            while process.returncode is None:
                if self._workspace_size(workspace) > maximum_workspace:
                    raise _WorkspaceLimitExceeded
                await asyncio.sleep(0.05)
            if self._workspace_size(workspace) > maximum_workspace:
                raise _WorkspaceLimitExceeded

        workspace_task = asyncio.create_task(monitor_workspace())
        try:
            stdout, stderr, _, _ = await asyncio.gather(
                stdout_task,
                stderr_task,
                wait_task,
                workspace_task,
            )
            return stdout, stderr
        except BaseException:
            for task in (stdout_task, stderr_task, wait_task, workspace_task):
                task.cancel()
            await asyncio.gather(
                stdout_task,
                stderr_task,
                wait_task,
                workspace_task,
                return_exceptions=True,
            )
            raise

    @staticmethod
    async def _kill(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await process.wait()

    @staticmethod
    def _decode(value: bytes) -> str:
        return value.decode("utf-8", errors="replace")

    @staticmethod
    def _workspace_size(workspace: Path) -> int:
        total = 0
        try:
            for path in workspace.rglob("*"):
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    total += path.stat().st_size
                except OSError:
                    continue
        except OSError:
            return total
        return total
