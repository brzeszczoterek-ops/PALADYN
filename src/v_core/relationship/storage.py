from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from .state import RelationshipState


class RelationshipStorage:

    def __init__(
        self,
        root: str = "memory/relationship",
    ):
        self.root = Path(root)
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path = self.root / "state.yaml"

    def save(
        self,
        state: RelationshipState,
    ) -> None:

        with self.path.open(
            "w",
            encoding="utf-8",
        ) as file:

            yaml.safe_dump(
                asdict(state),
                file,
                allow_unicode=True,
                sort_keys=False,
            )

    def load(self) -> RelationshipState:

        if not self.path.exists():
            return RelationshipState()

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = yaml.safe_load(file) or {}

        return RelationshipState(
            **data
        )
