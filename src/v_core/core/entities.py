from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4


class EntityType(str, Enum):
    PERSON = "person"
    AGENT = "agent"
    PERSONA = "persona"
    ORGANIZATION = "organization"
    PLACE = "place"
    DEVICE = "device"
    PROJECT = "project"
    CONCEPT = "concept"
    OBJECT = "object"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Entity:
    """
    A generic identifiable thing in the V-Core world model.

    Entity deliberately contains no persona-specific assumptions.
    A person can be related to one persona as a friend, to another
    as an operator, and to another as an unknown counterpart.
    """

    entity_id: UUID = field(
        default_factory=uuid4
    )

    entity_type: EntityType = (
        EntityType.UNKNOWN
    )

    name: str | None = None

    aliases: list[str] = field(
        default_factory=list
    )

    attributes: dict[str, object] = field(
        default_factory=dict
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    def add_alias(
        self,
        alias: str,
    ) -> None:

        alias = alias.strip()

        if not alias:
            return

        if alias not in self.aliases:
            self.aliases.append(alias)

    def set_attribute(
        self,
        key: str,
        value: object,
    ) -> None:

        if not key:
            return

        self.attributes[key] = value

    def get_attribute(
        self,
        key: str,
        default: object = None,
    ) -> object:

        return self.attributes.get(
            key,
            default,
        )

    def matches_name(
        self,
        value: str,
    ) -> bool:

        normalized = value.strip().casefold()

        if not normalized:
            return False

        candidates = [
            self.name,
            *self.aliases,
        ]

        return any(
            candidate
            and candidate.strip().casefold()
            == normalized
            for candidate in candidates
        )
