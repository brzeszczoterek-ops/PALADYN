from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import threading
import unicodedata
from typing import Any
from uuid import uuid4

from .models import SessionEvent


class Session:
    """Bounded conversation continuity, optionally durable across restarts.

    This ledger stores only the visible user message and V reply. Execution
    checkpoints, tool arguments, and model-authored memory policy deliberately
    stay out of it so a failed tool run cannot become conversational truth.
    """

    SCHEMA_VERSION = 1
    JOURNAL_NAME = "dialogue.jsonl"
    MAX_FIELD_CHARACTERS = 131_072

    def __init__(
        self,
        root: Path | None = None,
        *,
        max_events: int = 400,
        max_file_bytes: int = 4 * 1024 * 1024,
    ):
        self.events: list[SessionEvent] = []
        self.max_events = max(1, int(max_events))
        self.max_file_bytes = max(64 * 1024, int(max_file_bytes))
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.path = self.root / self.JOURNAL_NAME if self.root is not None else None
        self._lock = threading.Lock()
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
            self._load()

    @staticmethod
    def _safe_text(value: Any) -> str:
        return str(value or "")[: Session.MAX_FIELD_CHARACTERS]

    @classmethod
    def _safe_event(cls, event: SessionEvent) -> SessionEvent:
        return SessionEvent(
            timestamp=cls._safe_text(event.timestamp)[:128],
            event_type="task",
            data={
                "task": cls._safe_text(event.data.get("task", "")),
                "result": cls._safe_text(event.data.get("result", "")),
            },
        )

    @classmethod
    def _decode_event(cls, payload: Any) -> SessionEvent | None:
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            return None
        if payload.get("event_type") != "task":
            return None
        data = payload.get("data", {})
        if not isinstance(data, dict):
            return None
        task = data.get("task", "")
        result = data.get("result", "")
        timestamp = payload.get("timestamp", "")
        if not all(isinstance(value, str) for value in (task, result, timestamp)):
            return None
        if not task and not result:
            return None
        return cls._safe_event(
            SessionEvent(
                timestamp=timestamp,
                event_type="task",
                data={"task": task, "result": result},
            )
        )

    @classmethod
    def _encode_event(cls, event: SessionEvent) -> str:
        safe = cls._safe_event(event)
        return json.dumps(
            {
                "schema_version": cls.SCHEMA_VERSION,
                "timestamp": safe.timestamp,
                "event_type": safe.event_type,
                "data": safe.data,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            os.chmod(self.path, 0o600)
            size = self.path.stat().st_size
            with self.path.open("rb") as handle:
                if size > self.max_file_bytes:
                    handle.seek(size - self.max_file_bytes)
                    handle.readline()
                raw_lines = handle.readlines()
        except OSError:
            return

        loaded: deque[SessionEvent] = deque(maxlen=self.max_events)
        for raw_line in raw_lines:
            if not raw_line or len(raw_line) > 2 * self.MAX_FIELD_CHARACTERS + 4_096:
                continue
            try:
                payload = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            event = self._decode_event(payload)
            if event is not None:
                loaded.append(event)
        self.events = list(loaded)
        if size > self.max_file_bytes:
            self._compact()

    def _append(self, event: SessionEvent) -> None:
        if self.path is None:
            return
        encoded = self._encode_event(event).encode("utf-8")
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o600,
            )
            try:
                written = 0
                while written < len(encoded):
                    written += os.write(descriptor, encoded[written:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(self.path, 0o600)
            if self.path.stat().st_size > self.max_file_bytes:
                self._compact()
        except OSError:
            return

    def _compact(self) -> None:
        if self.path is None:
            return
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                for event in self.events[-self.max_events :]:
                    encoded = self._encode_event(event).encode("utf-8")
                    written = 0
                    while written < len(encoded):
                        written += os.write(descriptor, encoded[written:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def add(self, event_type: str, data: dict) -> None:
        event = SessionEvent(event_type=event_type, data=dict(data))
        if event_type != "task":
            self.events.append(event)
            return
        safe = self._safe_event(event)
        with self._lock:
            self.events.append(safe)
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events :]
            self._append(safe)

    def clear(self) -> None:
        with self._lock:
            self.events.clear()
            if self.path is not None:
                self._compact()

    def export(self) -> list[dict]:
        return [asdict(event) for event in self.events]

    @staticmethod
    def _match_tokens(text: str) -> set[str]:
        normalized = unicodedata.normalize("NFKD", text.casefold())
        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )
        return {
            token[:7]
            for token in re.findall(r"[^\W_]+", normalized, re.UNICODE)
            if len(token) >= 4
        }

    def _task_events(self) -> list[SessionEvent]:
        return [event for event in self.events if event.event_type == "task"]

    @staticmethod
    def _bounded_messages(
        events: list[SessionEvent],
        *,
        limit: int,
        max_characters: int | None,
    ) -> list[dict]:
        selected: list[SessionEvent] = []
        used_characters = 0
        for event in reversed(events):
            task = str(event.data.get("task", ""))
            result = str(event.data.get("result", ""))
            event_characters = len(task) + len(result)
            if (
                max_characters is not None
                and selected
                and used_characters + event_characters > max_characters
            ):
                continue
            selected.append(event)
            used_characters += event_characters
            if len(selected) >= limit:
                break

        history: list[dict] = []
        for event in reversed(selected):
            task = event.data.get("task", "")
            result = event.data.get("result", "")
            if task:
                history.append({"role": "user", "content": task})
            if result:
                history.append({"role": "assistant", "content": result})
        return history

    def messages(
        self,
        limit: int = 10,
        max_characters: int | None = None,
    ) -> list[dict]:
        return self._bounded_messages(
            self._task_events(),
            limit=limit,
            max_characters=max_characters,
        )

    def context_messages(
        self,
        prompt: str,
        *,
        limit: int = 6,
        max_characters: int | None = None,
    ) -> list[dict]:
        """Select recent turns plus older user requests relevant to ``prompt``."""

        events = self._task_events()
        if not events:
            return []
        recent_start = max(0, len(events) - min(3, limit))
        chosen = set(range(recent_start, len(events)))
        query_tokens = self._match_tokens(prompt)
        ranked: list[tuple[int, int]] = []
        for index, event in enumerate(events[:recent_start]):
            task_tokens = self._match_tokens(str(event.data.get("task", "")))
            score = len(query_tokens & task_tokens)
            if score:
                ranked.append((score, index))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, index in ranked:
            if len(chosen) >= limit:
                break
            chosen.add(index)
        for index in range(recent_start - 1, -1, -1):
            if len(chosen) >= limit:
                break
            chosen.add(index)
        selected = [events[index] for index in sorted(chosen)]
        return self._bounded_messages(
            selected,
            limit=limit,
            max_characters=max_characters,
        )

    def __len__(self) -> int:
        return len(self.events)
