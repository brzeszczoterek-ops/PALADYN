from __future__ import annotations

from uuid import uuid4

from .models import (
    ExperienceEntry,
    KnowledgeEntry,
    ReflectionEntry,
    SummaryEntry,
)
from .storage import MemoryStorage


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
            if entry.confidence < 0.35:
                return None

        if isinstance(entry, SummaryEntry):
            if not entry.summary and not entry.lessons:
                return None
            if entry.confidence < 0.50:
                return None

        if isinstance(entry, KnowledgeEntry):
            if not entry.title and not entry.content:
                return None
            if entry.confidence < 0.65:
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
