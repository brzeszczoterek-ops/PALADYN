from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .journal import TaskJournal
from .models import utc_now


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


@dataclass(slots=True)
class AgentTaskTrace:
    """Durable evidence for one interactive agent task.

    This is deliberately separate from model-authored conversation. Only the
    runtime can add tool checkpoints, so a model cannot turn a promise into an
    apparent execution record by printing JSON in its final answer.
    """

    root: Path
    objective: str
    task_id: str = field(default_factory=lambda: f"interactive-{uuid4().hex}")
    status: str = "running"
    started_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    requirements: dict[str, bool] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    _journal: TaskJournal = field(init=False, repr=False)
    _checkpoint_root: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self._journal = TaskJournal(self.root / "journal")
        self._checkpoint_root = self.root / "checkpoints"
        self._checkpoint_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self._checkpoint_root, 0o700)
        self._journal.append(
            self.task_id,
            "task_started",
            {"objective": self.objective},
        )
        self._save()

    def record_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        self.updated_at = utc_now()
        self._journal.append(self.task_id, event, data)
        self._save()

    def set_requirements(self, requirements: dict[str, bool]) -> None:
        self.requirements = dict(requirements)
        self.record_event("task_contract_created", {"requirements": self.requirements})

    def tool_started(self, name: str, arguments: dict | str) -> int:
        index = len(self.tool_calls) + 1
        call = {
            "sequence": index,
            "tool": name,
            "arguments": _json_safe(arguments),
            "status": "running",
            "started_at": utc_now(),
        }
        self.tool_calls.append(call)
        self.updated_at = utc_now()
        self._journal.append(
            self.task_id,
            "tool_started",
            call,
        )
        self._save()
        return index

    def tool_finished(
        self,
        sequence: int,
        result: str,
        *,
        error: str | None = None,
    ) -> None:
        call = self.tool_calls[sequence - 1]
        call["status"] = "failed" if error else "succeeded"
        call["finished_at"] = utc_now()
        call["result_sha256"] = _digest(result)
        call["result_excerpt"] = result[:2_000]
        if error:
            call["error"] = error
        self.updated_at = utc_now()
        self._journal.append(
            self.task_id,
            "tool_failed" if error else "tool_completed",
            call,
        )
        self._save()

    def complete(self, answer: str) -> None:
        self.status = "completed"
        self.updated_at = utc_now()
        self.finished_at = self.updated_at
        result = {
            "result_sha256": _digest(answer),
            "result_excerpt": answer[:2_000],
        }
        self._journal.append(self.task_id, "task_completed", result)
        self._save(result=result)

    def stop(self, reason: str) -> None:
        self._terminate("stopped", "task_stopped", reason)

    def fail(self, reason: str) -> None:
        self._terminate("failed", "task_failed", reason)

    def block(self, reason: str) -> None:
        self._terminate("blocked", "task_blocked", reason)

    def evidence(self) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        for call in self.tool_calls:
            arguments = json.dumps(
                call.get("arguments", {}),
                ensure_ascii=False,
                default=str,
            )
            calls.append(
                {
                    "sequence": call.get("sequence"),
                    "tool": call.get("tool"),
                    "status": call.get("status"),
                    "arguments_sha256": _digest(arguments),
                    "arguments_excerpt": arguments[:2_000],
                    "result_sha256": call.get("result_sha256", ""),
                    "result_excerpt": call.get("result_excerpt", ""),
                    "error": call.get("error", ""),
                }
            )
        return {
            "task_id": self.task_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "tool_calls": calls,
            "requirements": self.requirements,
            "successful_tool_count": sum(
                call.get("status") == "succeeded" for call in self.tool_calls
            ),
            "failed_tool_count": sum(
                call.get("status") == "failed" for call in self.tool_calls
            ),
        }

    @staticmethod
    def latest_context(root: Path) -> dict[str, Any] | None:
        """Load a bounded runtime-authored summary for follow-up recovery."""

        checkpoints = Path(root) / "checkpoints"
        try:
            candidates = sorted(
                checkpoints.glob("interactive-*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            return None
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            calls = []
            for call in payload.get("tool_calls", [])[-6:]:
                calls.append(
                    {
                        "tool": str(call.get("tool", ""))[:128],
                        "status": str(call.get("status", ""))[:32],
                        "error": str(call.get("error", ""))[:500],
                    }
                )
            return {
                "task_id": str(payload.get("task_id", ""))[:128],
                "objective": str(payload.get("objective", ""))[:2_000],
                "status": str(payload.get("status", ""))[:32],
                "requirements": payload.get("requirements", {}),
                "tool_calls": calls,
            }
        return None

    def _save(self, *, result: dict[str, Any] | None = None) -> None:
        path = self._checkpoint_root / f"{self.task_id}.json"
        temporary = self._checkpoint_root / f".{self.task_id}.tmp"
        payload = {
            "schema_version": 2,
            "task_id": self.task_id,
            "objective": self.objective,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "requirements": self.requirements,
            "tool_calls": self.tool_calls,
        }
        if result is not None:
            payload["result"] = result

        with temporary.open("w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)

    def _terminate(self, status: str, event: str, reason: str) -> None:
        if self.status != "running":
            return
        self.status = status
        self.updated_at = utc_now()
        self.finished_at = self.updated_at
        data = {"reason": reason[:2_000]}
        self._journal.append(self.task_id, event, data)
        self._save(result=data)
