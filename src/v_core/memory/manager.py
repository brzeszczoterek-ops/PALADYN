from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
from typing import Any, Iterable
from uuid import uuid4

from .models import (
    ExperienceEntry,
    KnowledgeEntry,
    ReflectionEntry,
    SummaryEntry,
)
from .storage import MemoryStorage


def clip_text(value: object, max_chars: int) -> str:
    """Bound untrusted memory evidence without losing both ends of the text."""

    text = str(value or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    marker = "\n...[memory evidence clipped]...\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    available = max_chars - len(marker)
    head = max(1, (available * 2) // 3)
    tail = max(0, available - head)
    return f"{text[:head]}{marker}{text[-tail:] if tail else ''}"


def recent_records_json(
    records: Iterable[Any],
    *,
    max_records: int = 12,
    max_chars: int = 3_000,
) -> str:
    """Render the newest bounded records as JSON for a memory-stage prompt."""

    normalized = [_plain(record) for record in records]
    normalized.sort(key=_timestamp, reverse=True)
    candidates = normalized[: max(0, max_records)]
    selected: list[Any] = []

    for record in candidates:
        candidate = {
            "records_newest_first": [*selected, record],
            "omitted_older_records": len(normalized) - len(selected) - 1,
        }
        if len(_json(candidate)) > max_chars:
            break
        selected.append(record)

    payload: dict[str, Any] = {
        "records_newest_first": selected,
        "omitted_older_records": len(normalized) - len(selected),
    }
    rendered = _json(payload)
    if selected or not normalized or len(rendered) <= max_chars:
        return rendered

    # A single hostile or malformed record may itself be enormous. Keep a
    # bounded excerpt as a JSON string rather than invalid partial JSON.
    latest = _json(normalized[0])
    payload = {
        "latest_record_excerpt": clip_text(latest, max(0, max_chars - 160)),
        "omitted_older_records": len(normalized) - 1,
    }
    rendered = _json(payload)
    while len(rendered) > max_chars and payload["latest_record_excerpt"]:
        excess = len(rendered) - max_chars
        excerpt = payload["latest_record_excerpt"]
        payload["latest_record_excerpt"] = excerpt[
            : max(0, len(excerpt) - excess - 1)
        ]
        rendered = _json(payload)
    return rendered


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _timestamp(record: Any) -> str:
    if isinstance(record, dict):
        return str(record.get("timestamp", ""))
    return ""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


class MemoryManager:

    def __init__(
        self,
        storage: MemoryStorage,
    ):
        self.storage = storage

    def remember(
        self,
        category: str,
        entry,
    ):
        """
        Persist a memory entry.

        Entries explicitly marked as non-reusable reflections are not
        persisted as long-term memory.
        """

        if isinstance(entry, ReflectionEntry):
            if not entry.remember:
                return None

        if isinstance(entry, ExperienceEntry):
            if not entry.summary and not entry.lesson:
                return None
            if not self._reliable(entry.confidence, 0.35):
                return None

        if isinstance(entry, SummaryEntry):
            if not entry.summary and not entry.lessons:
                return None
            if not self._reliable(entry.confidence, 0.50):
                return None

        if isinstance(entry, KnowledgeEntry):
            if not entry.title and not entry.content:
                return None
            if not self._reliable(entry.confidence, 0.65):
                return None

        filename = f"{uuid4().hex}.yaml"

        return self.storage.save(
            category,
            filename,
            entry,
        )

    def load_all(
        self,
        category: str,
    ):

        items = []

        for file in self.storage.list(category):
            items.append(
                self.storage.load(file)
            )

        return items

    def latest(
        self,
        category: str,
    ):

        files = self.storage.list(category)

        if not files:
            return None

        latest_file = max(
            files,
            key=lambda path: path.stat().st_mtime,
        )

        return self.storage.load(
            latest_file
        )

    def count(
        self,
        category: str,
    ) -> int:

        return len(
            self.storage.list(category)
        )

    @staticmethod
    def _reliable(value: object, minimum: float) -> bool:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(confidence) and confidence >= minimum
