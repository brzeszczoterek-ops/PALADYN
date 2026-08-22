from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from .entities import Entity
from .relationships import Relationship
from .observations import Observation


@dataclass(slots=True)
class CoreState:
    """
    Persona-agnostic state of the V-Core world model.

    This object does not know anything about V, Boss, or any
    specific persona. It represents the shared domain state on
    which higher-level systems operate.
    """

    entities: dict[UUID, Entity] = field(
        default_factory=dict
    )

    relationships: dict[UUID, Relationship] = field(
        default_factory=dict
    )

    observations: list[Observation] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    def add_entity(
        self,
        entity: Entity,
    ) -> None:

        self.entities[
            entity.entity_id
        ] = entity

        self.touch()

    def get_entity(
        self,
        entity_id: UUID,
    ) -> Entity | None:

        return self.entities.get(
            entity_id
        )

    def add_relationship(
        self,
        relationship: Relationship,
    ) -> None:

        self.relationships[
            relationship.relationship_id
        ] = relationship

        self.touch()

    def get_relationship(
        self,
        relationship_id: UUID,
    ) -> Relationship | None:

        return self.relationships.get(
            relationship_id
        )

    def add_observation(
        self,
        observation: Observation,
    ) -> None:

        self.observations.append(
            observation
        )

        self.touch()

    def touch(self) -> None:

        self.updated_at = datetime.now(
            timezone.utc
        )
