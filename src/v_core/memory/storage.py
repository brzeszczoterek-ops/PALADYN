from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import yaml


class MemoryStorage:

    def __init__(self, root: Path):

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
