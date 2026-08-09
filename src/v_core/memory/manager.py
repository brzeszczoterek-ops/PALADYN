from __future__ import annotations

from uuid import uuid4

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

        return self.storage.load(
            files[-1]
        )

    def count(
        self,
        category: str,
    ) -> int:

        return len(
            self.storage.list(category)
        )
