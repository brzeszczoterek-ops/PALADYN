from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import AuthorizationEnvelope, AutonomousTask, utc_now


_TASK_ID = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def validate_task_id(task_id: str) -> str:
    if not _TASK_ID.fullmatch(task_id):
        raise ValueError("invalid task_id")
    return task_id


@dataclass(slots=True)
class TaskJournal:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def append(
        self,
        task_id: str,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> Path:
        validate_task_id(task_id)
        path = self.root / f"{task_id}.jsonl"
        record = {
            "timestamp": utc_now(),
            "task_id": task_id,
            "event": event,
            "data": data or {},
        }

        with path.open("a", encoding="utf-8") as handle:
            os.chmod(path, 0o600)
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        return path

    def read(self, task_id: str) -> list[dict[str, Any]]:
        validate_task_id(task_id)
        path = self.root / f"{task_id}.jsonl"
        if not path.exists():
            return []

        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


@dataclass(slots=True)
class CheckpointStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def save(
        self,
        task: AutonomousTask,
        envelope: AuthorizationEnvelope,
    ) -> Path:
        validate_task_id(task.task_id)
        path = self.root / f"{task.task_id}.json"
        temporary = self.root / f".{task.task_id}.tmp"
        payload = {
            "schema_version": 1,
            "task": task.to_dict(),
            "authorization_envelope": envelope.to_dict(),
        }

        with temporary.open("w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, path)
        os.chmod(path, 0o600)
        return path

    def load(
        self,
        task_id: str,
    ) -> tuple[AutonomousTask, AuthorizationEnvelope] | None:
        validate_task_id(task_id)
        path = self.root / f"{task_id}.json"
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        return (
            AutonomousTask.from_dict(payload["task"]),
            AuthorizationEnvelope.from_dict(payload["authorization_envelope"]),
        )
