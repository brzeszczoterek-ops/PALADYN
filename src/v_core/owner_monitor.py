from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from typing import Callable
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

from .model_loader import LlamaServerSession


_TRUTHY = {"1", "true", "yes", "on"}
_PROMPT_TIMING = re.compile(
    r"prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens.*?"
    r"([\d.]+) tokens per second"
)
_GENERATION_TIMING = re.compile(
    r"(?<!prompt )eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens.*?"
    r"([\d.]+) tokens per second"
)
_TOTAL_TIMING = re.compile(
    r"total time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens"
)
_DIRECT_OPENER = build_opener(ProxyHandler({}))


@dataclass(frozen=True, slots=True)
class ResponseTiming:
    prompt_tokens: int | None = None
    prompt_tokens_per_second: float | None = None
    generated_tokens: int | None = None
    generated_tokens_per_second: float | None = None
    total_milliseconds: float | None = None


@dataclass(frozen=True, slots=True)
class MonitorTarget:
    pid: int
    port: int
    model: str
    context_size: int
    reasoning: str
    cache_type_k: str
    cache_type_v: str
    log_path: Path
    session_id: str
    telemetry_path: Path


class TelemetryJournal:
    """Append-only, private telemetry for exactly one model session."""

    def __init__(self, path: Path, session_id: str):
        self.path = Path(path).resolve()
        self.session_id = session_id
        self._handle = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.parent.chmod(0o700)
            self._handle = self.path.open("a", encoding="utf-8", buffering=1)
            self.path.chmod(0o600)
        except OSError:
            self._handle = None

    def append(self, event: str, data: dict[str, object]) -> None:
        if self._handle is None:
            return
        payload = {
            "schema_version": 1,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "data": data,
        }
        try:
            self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            self.close()

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None


def monitor_session_identity(session: LlamaServerSession) -> tuple[str, Path]:
    stamp = session.log_path.stem.removeprefix("llama-server-")
    session_id = f"{stamp}-pid{session.process.pid}"
    telemetry = (
        session.log_path.parent.parent
        / "monitor_sessions"
        / f"{session_id}.jsonl"
    )
    return session_id, telemetry


def parse_prometheus(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            values[name] = values.get(name, 0.0) + float(fields[-1])
        except ValueError:
            continue
    return values


def parse_response_timing(text: str) -> ResponseTiming:
    prompt = list(_PROMPT_TIMING.finditer(text))
    generation = list(_GENERATION_TIMING.finditer(text))
    total = list(_TOTAL_TIMING.finditer(text))
    return ResponseTiming(
        prompt_tokens=int(prompt[-1].group(1)) if prompt else None,
        prompt_tokens_per_second=float(prompt[-1].group(2)) if prompt else None,
        generated_tokens=int(generation[-1].group(1)) if generation else None,
        generated_tokens_per_second=(
            float(generation[-1].group(2)) if generation else None
        ),
        total_milliseconds=float(total[-1].group(1)) if total else None,
    )


def counter_rate(
    current: dict[str, float],
    previous: dict[str, float],
    *,
    token_suffix: str,
    seconds_suffix: str,
) -> float | None:
    current_tokens = _metric(current, token_suffix)
    previous_tokens = _metric(previous, token_suffix)
    current_seconds = _metric(current, seconds_suffix)
    previous_seconds = _metric(previous, seconds_suffix)
    if None in {
        current_tokens,
        previous_tokens,
        current_seconds,
        previous_seconds,
    }:
        return None
    token_delta = float(current_tokens) - float(previous_tokens)
    seconds_delta = float(current_seconds) - float(previous_seconds)
    if token_delta <= 0 or seconds_delta <= 0:
        return None
    return token_delta / seconds_delta


def compact_tegrastats(line: str) -> tuple[str, str, str]:
    ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    swap = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
    cpu_block = re.search(r"CPU\s+\[([^]]+)]", line)
    gpu_load = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    cpu_temp = re.search(r"cpu@([\d.]+)C", line)
    gpu_temp = re.search(r"gpu@([\d.]+)C", line)
    power = re.search(r"VDD_IN\s+(\d+)mW", line)

    cpu_values = (
        [int(value) for value in re.findall(r"(\d+)%", cpu_block.group(1))]
        if cpu_block
        else []
    )
    cpu_average = sum(cpu_values) / len(cpu_values) if cpu_values else None
    memory_line = (
        f"RAM {ram.group(1)}/{ram.group(2)} MB" if ram else "RAM —"
    )
    if swap:
        memory_line += f"  ·  SWAP {swap.group(1)}/{swap.group(2)} MB"
    compute_line = (
        f"CPU {'—' if cpu_average is None else f'{cpu_average:.0f}%'}"
        f"  ·  GPU {'—' if gpu_load is None else gpu_load.group(1) + '%'}"
        f"  ·  temp CPU {'—' if cpu_temp is None else cpu_temp.group(1) + '°C'}"
        f" / GPU {'—' if gpu_temp is None else gpu_temp.group(1) + '°C'}"
    )
    power_line = (
        "Power —" if power is None else f"Power {int(power.group(1)) / 1000:.2f} W"
    )
    return memory_line, compute_line, power_line


def build_monitor_command(
    session: LlamaServerSession,
    *,
    terminal: Path,
    python: Path,
) -> tuple[str, ...]:
    profile = session.profile
    session_id, telemetry_path = monitor_session_identity(session)
    return (
        str(terminal),
        "--window",
        "--geometry=92x31",
        "--title=V Owner Monitor",
        "--",
        str(python),
        "-m",
        "v_core.owner_monitor",
        "--pid",
        str(session.process.pid),
        "--port",
        str(profile.port),
        "--model",
        profile.alias,
        "--context",
        str(profile.context_size),
        "--reasoning",
        profile.reasoning,
        "--cache-k",
        profile.cache_type_k,
        "--cache-v",
        profile.cache_type_v,
        "--log",
        str(session.log_path),
        "--session",
        session_id,
        "--telemetry",
        str(telemetry_path),
    )


def launch_owner_monitor(
    session: LlamaServerSession | None,
    *,
    environ: dict[str, str] | None = None,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> bool:
    environment = os.environ if environ is None else environ
    enabled = environment.get("PALADYN_OWNER_MONITOR", "0").strip().casefold()
    if session is None or enabled not in _TRUTHY:
        return False
    configured = environment.get("PALADYN_OWNER_TERMINAL", "").strip()
    discovered = configured or shutil.which("gnome-terminal")
    if not discovered:
        return False
    command = build_monitor_command(
        session,
        terminal=Path(discovered),
        python=Path(sys.executable),
    )
    try:
        popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            env=dict(environment),
        )
    except OSError:
        return False
    return True


def _http_text(url: str) -> str:
    try:
        with _DIRECT_OPENER.open(url, timeout=0.6) as response:
            return response.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError, OSError):
        return ""


def _tail(path: Path, maximum: int = 128 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum))
            return handle.read(maximum).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _metric(metrics: dict[str, float], suffix: str) -> float | None:
    for name, value in metrics.items():
        if name.casefold().endswith(suffix.casefold()):
            return value
    return None


def _slot_summary(payload: str, fallback_context: int) -> tuple[str, int, int]:
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return "unavailable", 0, fallback_context
    if not isinstance(raw, list) or not raw:
        return "idle", 0, fallback_context
    slot = next(
        (
            item
            for item in raw
            if isinstance(item, dict)
            and str(item.get("state", "idle")).casefold() not in {"idle", "0"}
        ),
        raw[0],
    )
    if not isinstance(slot, dict):
        return "unavailable", 0, fallback_context
    state = str(slot.get("state", "idle"))
    used = int(slot.get("n_past", slot.get("n_prompt_tokens", 0)) or 0)
    context = int(slot.get("n_ctx", fallback_context) or fallback_context)
    return state, used, context


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f} tok/s"


def _format_integer(value: float | None) -> str:
    return "—" if value is None else f"{int(value):,}".replace(",", " ")


def run_monitor(target: MonitorTarget, *, interval: float = 1.0) -> None:
    tegra: subprocess.Popen[str] | None = None
    tegra_line = "unavailable"
    previous_metrics: dict[str, float] = {}
    record_interval = _environment_interval(
        "PALADYN_OWNER_MONITOR_RECORD_INTERVAL",
        5.0,
    )
    last_recorded = 0.0
    journal = TelemetryJournal(target.telemetry_path, target.session_id)
    journal.append(
        "session_start",
        {
            "model": target.model,
            "pid": target.pid,
            "port": target.port,
            "context_size": target.context_size,
            "reasoning": target.reasoning,
            "cache_type_k": target.cache_type_k,
            "cache_type_v": target.cache_type_v,
            "llama_log": str(target.log_path),
        },
    )
    tegra_binary = shutil.which("tegrastats")
    if tegra_binary:
        try:
            tegra = subprocess.Popen(
                [tegra_binary, "--interval", "1000"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
        except OSError:
            tegra = None

    try:
        while _alive(target.pid):
            if tegra is not None and tegra.stdout is not None:
                while select.select([tegra.stdout], [], [], 0)[0]:
                    line = tegra.stdout.readline().strip()
                    if line:
                        tegra_line = line

            base = f"http://127.0.0.1:{target.port}"
            metrics = parse_prometheus(_http_text(f"{base}/metrics"))
            state, used, context = _slot_summary(
                _http_text(f"{base}/slots"), target.context_size
            )
            timing = parse_response_timing(_tail(target.log_path))

            prompt_total = _metric(metrics, "prompt_tokens_total")
            generated_total = _metric(metrics, "tokens_predicted_total")
            live_prompt_rate = counter_rate(
                metrics,
                previous_metrics,
                token_suffix="prompt_tokens_total",
                seconds_suffix="prompt_tokens_seconds",
            )
            live_generation_rate = counter_rate(
                metrics,
                previous_metrics,
                token_suffix="tokens_predicted_total",
                seconds_suffix="tokens_predicted_seconds",
            )
            processing = _metric(metrics, "requests_processing")
            deferred = _metric(metrics, "requests_deferred")
            kv_usage = _metric(metrics, "kv_cache_usage_ratio")
            context_percent = (100.0 * used / context) if context else 0.0
            hardware = compact_tegrastats(tegra_line)
            now = time.monotonic()

            if now - last_recorded >= record_interval:
                journal.append(
                    "sample",
                    {
                        "slot_state": state,
                        "context_used": used,
                        "context_size": context,
                        "context_percent": round(context_percent, 4),
                        "kv_usage_ratio": kv_usage,
                        "requests_active": processing,
                        "requests_queued": deferred,
                        "live_prompt_tokens_per_second": live_prompt_rate,
                        "live_generation_tokens_per_second": live_generation_rate,
                        "prompt_tokens_total": prompt_total,
                        "generated_tokens_total": generated_total,
                        "last_response": {
                            "prompt_tokens": timing.prompt_tokens,
                            "prompt_tokens_per_second": timing.prompt_tokens_per_second,
                            "generated_tokens": timing.generated_tokens,
                            "generated_tokens_per_second": timing.generated_tokens_per_second,
                            "total_milliseconds": timing.total_milliseconds,
                        },
                        "hardware": {
                            "memory": hardware[0],
                            "compute": hardware[1],
                            "power": hardware[2],
                            "tegrastats_raw": tegra_line,
                        },
                    },
                )
                last_recorded = now

            lines = [
                "\033[2J\033[H",
                "V OWNER MONITOR  ·  local only",
                "=" * 54,
                f"Model       {target.model}",
                f"Reasoning   {target.reasoning}",
                f"KV cache    K={target.cache_type_k}  V={target.cache_type_v}",
                f"Server      pid={target.pid}  port={target.port}",
                f"Session     {target.session_id}",
                f"Saved       {target.telemetry_path.name}",
                "",
                f"Slot        {state}",
                f"Context     {used:,} / {context:,} tokens  ({context_percent:.1f}%)",
                f"KV usage    {'—' if kv_usage is None else f'{kv_usage * 100:.1f}%'}",
                f"Requests    active={_format_integer(processing)}  queued={_format_integer(deferred)}",
                "",
                "Live throughput",
                f"  prompt      {_format_rate(live_prompt_rate)}",
                f"  generation  {_format_rate(live_generation_rate)}",
                "",
                "Cumulative",
                f"  prompt tokens     {_format_integer(prompt_total)}",
                f"  generated tokens  {_format_integer(generated_total)}",
                "",
                "Last completed response",
                f"  prompt      {_format_integer(float(timing.prompt_tokens) if timing.prompt_tokens is not None else None)}  ·  {_format_rate(timing.prompt_tokens_per_second)}",
                f"  generation  {_format_integer(float(timing.generated_tokens) if timing.generated_tokens is not None else None)}  ·  {_format_rate(timing.generated_tokens_per_second)}",
                f"  total time  {'—' if timing.total_milliseconds is None else f'{timing.total_milliseconds / 1000:.2f} s'}",
                "",
                "Jetson",
                f"  {hardware[0]}",
                f"  {hardware[1]}",
                f"  {hardware[2]}",
                "",
                "This window closes when the local model stops.",
            ]
            print("\n".join(lines), end="", flush=True)
            previous_metrics = metrics
            time.sleep(max(0.2, interval))
    finally:
        journal.append("session_end", {"model_pid_alive": _alive(target.pid)})
        journal.close()
        if tegra is not None and tegra.poll() is None:
            try:
                os.killpg(tegra.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                tegra.wait(timeout=2)
            except subprocess.TimeoutExpired:
                tegra.kill()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PALADYN owner model monitor")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--reasoning", required=True)
    parser.add_argument("--cache-k", required=True)
    parser.add_argument("--cache-v", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    return parser


def _environment_interval(name: str, default: float) -> float:
    try:
        return max(1.0, min(300.0, float(os.getenv(name, str(default)))))
    except ValueError:
        return default


def main() -> None:
    args = _parser().parse_args()
    run_monitor(
        MonitorTarget(
            pid=args.pid,
            port=args.port,
            model=args.model,
            context_size=args.context,
            reasoning=args.reasoning,
            cache_type_k=args.cache_k,
            cache_type_v=args.cache_v,
            log_path=args.log,
            session_id=args.session,
            telemetry_path=args.telemetry,
        )
    )


if __name__ == "__main__":
    main()
