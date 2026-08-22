from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import yaml


class MemoryStorage:

    def __init__(
        self,
        root: Path,
    ):

        self.root = Path(root)

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        category: str,
        filename: str,
        data,
    ) -> Path:

        directory = self.root / category

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = directory / filename

        if is_dataclass(data):
            data = asdict(data)

        data = self._serialize(
            data
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            yaml.safe_dump(
                data,
                f,
                allow_unicode=True,
                sort_keys=False,
            )

        return path

    def load(
        self,
        path: Path,
    ) -> dict:

        if not path.exists():
            return {}

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return yaml.safe_load(f) or {}

    def list(
        self,
        category: str,
    ) -> list[Path]:

        directory = self.root / category

        if not directory.exists():
            return []

        return sorted(
            directory.glob("*.yaml")
        )

    @classmethod
    def _serialize(
        cls,
        value,
    ):
        if isinstance(value, Enum):
            return value.value

        if isinstance(value, dict):
            return {
                key: cls._serialize(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                cls._serialize(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                cls._serialize(item)
                for item in value
            ]

        if isinstance(value, set):
            return [
                cls._serialize(item)
                for item in value
            ]

        return value
