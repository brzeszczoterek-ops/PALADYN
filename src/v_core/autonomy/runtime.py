from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import signal
from typing import Any


def _process_start_ticks(pid: int) -> int | None:
    try:
        content = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None

    # The executable name in field 2 may contain spaces and parentheses. The
    # final ')' is the only safe split point before fields 3..N.
    closing = content.rfind(")")
    if closing < 0:
        return None
    fields = content[closing + 2 :].split()
    try:
        return int(fields[19])  # proc(5) field 22, starttime
    except (IndexError, ValueError):
        return None


@dataclass(slots=True)
class RuntimeRegistry:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, name: str, *, pid: int | None = None) -> Path:
        process_id = pid if pid is not None else os.getpid()
        start_ticks = _process_start_ticks(process_id)
        if start_ticks is None:
            raise RuntimeError(f"cannot inspect process {process_id}")
        payload = {
            "schema_version": 1,
            "name": name,
            "pid": process_id,
            "start_ticks": start_ticks,
            "registered_at": datetime.now(UTC).isoformat(),
        }
        path = self.root / f"{process_id}.json"
        temporary = self.root / f".{process_id}.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return path

    def unregister(self, *, pid: int | None = None) -> None:
        process_id = pid if pid is not None else os.getpid()
        path = self.root / f"{process_id}.json"
        if not path.exists():
            return
        record = self._read(path)
        if record and self._matches_process(record):
            path.unlink()

    def active(self) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            record = self._read(path)
            if record and self._matches_process(record):
                active.append(record)
            else:
                path.unlink(missing_ok=True)
        return active

    def terminate_all(self, *, sig: signal.Signals = signal.SIGTERM) -> list[int]:
        terminated: list[int] = []
        own_pid = os.getpid()
        for record in self.active():
            pid = int(record["pid"])
            if pid == own_pid:
                continue
            try:
                os.kill(pid, sig)
            except (PermissionError, ProcessLookupError):
                continue
            terminated.append(pid)
        return terminated

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _matches_process(record: dict[str, Any]) -> bool:
        try:
            pid = int(record["pid"])
            expected = int(record["start_ticks"])
        except (KeyError, TypeError, ValueError):
            return False
        return _process_start_ticks(pid) == expected
