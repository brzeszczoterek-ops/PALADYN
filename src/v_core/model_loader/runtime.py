from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Callable

import httpx

from .models import ModelProfile


class LlamaServerUnavailable(RuntimeError):
    pass


class LlamaServerStartError(RuntimeError):
    pass


def find_llama_server(
    configured: str = "",
    *,
    environ: dict[str, str] | None = None,
) -> Path | None:
    environment = environ if environ is not None else os.environ
    candidates: list[Path] = []
    explicit = environment.get("LLAMA_CPP_SERVER", "").strip() or configured.strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    discovered = shutil.which("llama-server")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(
        [
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
            Path.home() / ".local" / "bin" / "llama-server",
            Path("/usr/local/bin/llama-server"),
            Path("/usr/bin/llama-server"),
        ]
    )
    try:
        candidates.extend(
            sorted(Path.home().glob("llama*/build/bin/llama-server"))[:64]
        )
    except OSError:
        pass
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    return None


def build_server_command(binary: Path, profile: ModelProfile) -> tuple[str, ...]:
    model = Path(profile.model_path).resolve(strict=True)
    executable = Path(binary).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise LlamaServerUnavailable("llama-server is not executable")
    if not model.is_file() or model.suffix.casefold() != ".gguf":
        raise LlamaServerUnavailable("selected GGUF model no longer exists")
    try:
        with model.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise LlamaServerUnavailable("selected model has no GGUF header")
    except OSError as error:
        raise LlamaServerUnavailable("selected GGUF model cannot be read") from error
    command = [
        str(executable),
        "--model",
        str(model),
        "--alias",
        profile.alias,
        "--port",
        str(profile.port),
        "--ctx-size",
        str(profile.context_size),
        "--n-gpu-layers",
        profile.gpu_layers,
        "--batch-size",
        str(profile.batch_size),
        "--ubatch-size",
        str(profile.ubatch_size),
        "--parallel",
        str(profile.parallel),
        "--flash-attn",
        profile.flash_attention,
        "--reasoning",
        profile.reasoning,
        "--cache-type-k",
        profile.cache_type_k,
        "--cache-type-v",
        profile.cache_type_v,
    ]
    anti_repetition = {
        "off": (),
        "balanced": (
            "--repeat-last-n",
            "256",
            "--repeat-penalty",
            "1.08",
            "--dry-multiplier",
            "0.8",
            "--dry-base",
            "1.75",
            "--dry-allowed-length",
            "2",
            "--dry-penalty-last-n",
            "2048",
        ),
        "strong": (
            "--repeat-last-n",
            "512",
            "--repeat-penalty",
            "1.12",
            "--dry-multiplier",
            "1.1",
            "--dry-base",
            "1.75",
            "--dry-allowed-length",
            "2",
            "--dry-penalty-last-n",
            "4096",
        ),
    }[profile.anti_repetition]
    command.extend(anti_repetition)
    if profile.threads:
        command.extend(("--threads", str(profile.threads)))
    command.extend(profile.extra_args)
    # Enforced arguments are deliberately last so a profile cannot enable
    # downloads, llama.cpp's own filesystem tools, or a public listener.
    command.extend(
        (
            "--metrics",
            "--slots",
            "--offline",
            "--no-webui",
            "--host",
            "127.0.0.1",
        )
    )
    return tuple(command)


@dataclass(slots=True)
class LlamaServerSession:
    profile: ModelProfile
    process: subprocess.Popen[bytes]
    log_path: Path
    _log_handle: object

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.profile.port}/v1"

    def apply_to_environment(self) -> None:
        os.environ["V_CORE_PROVIDER"] = "llama_cpp"
        os.environ["V_CORE_BASE_URL"] = self.base_url
        os.environ["V_CORE_MODEL"] = self.profile.alias
        os.environ["V_CORE_CONTEXT"] = str(self.profile.context_size)
        os.environ["V_CORE_TEMPERATURE"] = str(self.profile.temperature)
        os.environ["V_CORE_TOP_P"] = str(self.profile.top_p)

    async def stop(self, *, timeout: float = 10.0) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.to_thread(self.process.wait, timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(self.process.wait)
        close = getattr(self._log_handle, "close", None)
        if callable(close):
            close()


async def start_llama_server(
    binary: Path,
    profile: ModelProfile,
    runtime_root: Path,
    *,
    status: Callable[[str], None] | None = None,
) -> LlamaServerSession:
    output = status or (lambda _: None)
    root = Path(runtime_root).resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logs.chmod(0o700)

    if await _endpoint_responds(profile.port):
        raise LlamaServerStartError(
            f"port {profile.port} already has an HTTP service; refusing to replace it"
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = logs / f"llama-server-{timestamp}.log"
    command = build_server_command(binary, profile)
    log_handle = log_path.open("ab", buffering=0)
    log_path.chmod(0o600)
    try:
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("LLAMA_ARG_")
        }
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )
    except OSError as error:
        log_handle.close()
        raise LlamaServerStartError(f"could not start llama-server: {error}") from error

    session = LlamaServerSession(profile, process, log_path, log_handle)
    output(f"Loading {Path(profile.model_path).name} with llama.cpp …")
    deadline = time.monotonic() + profile.startup_timeout_seconds
    last_message = "waiting for /health"
    try:
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            while time.monotonic() < deadline:
                return_code = process.poll()
                if return_code is not None:
                    raise LlamaServerStartError(
                        f"llama-server exited with code {return_code}: "
                        f"{_tail(log_path)}"
                    )
                try:
                    response = await client.get(
                        f"http://127.0.0.1:{profile.port}/health"
                    )
                    if response.status_code == 200:
                        payload = response.json()
                        if isinstance(payload, dict) and payload.get("status") == "ok":
                            await _verify_model_alias(client, profile)
                            session.apply_to_environment()
                            output(f"Model {profile.alias} is ready.")
                            return session
                    if response.status_code == 503:
                        try:
                            payload = response.json()
                            last_message = json.dumps(payload, ensure_ascii=False)[:500]
                        except ValueError:
                            last_message = response.text[:500]
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(0.25)
        raise LlamaServerStartError(
            f"llama-server did not become ready within "
            f"{profile.startup_timeout_seconds:g}s ({last_message})"
        )
    except BaseException:
        await session.stop()
        raise


async def _endpoint_responds(port: int) -> bool:
    try:
        async with httpx.AsyncClient(timeout=0.5, trust_env=False) as client:
            response = await client.get(f"http://127.0.0.1:{port}/health")
            return response.status_code in {200, 401, 403, 404, 503}
    except httpx.HTTPError:
        return False


async def _verify_model_alias(
    client: httpx.AsyncClient,
    profile: ModelProfile,
) -> None:
    response = await client.get(f"http://127.0.0.1:{profile.port}/v1/models")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise LlamaServerStartError("llama-server returned an invalid model list")
    identifiers = {
        str(item.get("id"))
        for item in payload["data"]
        if isinstance(item, dict)
    }
    if profile.alias not in identifiers:
        raise LlamaServerStartError(
            f"llama-server health passed but alias {profile.alias!r} is absent"
        )


def _tail(path: Path, maximum: int = 4_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum))
            return handle.read(maximum).decode("utf-8", errors="replace").strip()
    except OSError:
        return "log unavailable"
