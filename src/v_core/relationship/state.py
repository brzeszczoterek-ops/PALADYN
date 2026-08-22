from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RelationshipState:

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    familiarity: float = 0.0

    trust: float = 0.0

    emotional_bond: float = 0.0

    relationship_depth: float = 0.0

    understanding_of_boss: float = 0.0

    shared_history: list[str] = field(
        default_factory=list
    )

    preferred_forms_of_address: list[str] = field(
        default_factory=list
    )

    def touch(self) -> None:

        self.updated_at = datetime.now().isoformat()

    def add_shared_experience(
        self,
        experience: str,
    ) -> None:

        if not experience:
            return

        if experience not in self.shared_history:

            self.shared_history.append(
                experience
            )

            self.touch()

    def add_form_of_address(
        self,
        form: str,
    ) -> None:

        if not form:
            return

        if form not in self.preferred_forms_of_address:

            self.preferred_forms_of_address.append(
                form
            )

            self.touch()
