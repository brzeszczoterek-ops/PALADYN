from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

from .provenance import Provenance


class RelationshipType(str, Enum):
    UNKNOWN = "unknown"

    COLLABORATOR = "collaborator"
    FRIEND = "friend"
    FAMILY = "family"
    OPERATOR = "operator"
    CREATOR = "creator"
    USER = "user"
    COLLEAGUE = "colleague"

    MEMBER_OF = "member_of"
    OWNS = "owns"
    USES = "uses"
    LOCATED_AT = "located_at"
    RELATED_TO = "related_to"
    CUSTOM = "custom"


@dataclass(slots=True)
class Relationship:
    """
    A relationship between two entities.

    The relationship belongs to neither entity exclusively.
    It is a separate object and may evolve over time.
    """

    relationship_id: UUID = field(
        default_factory=uuid4
    )

    source_entity_id: UUID | None = None

    target_entity_id: UUID | None = None

    relationship_type: RelationshipType = (
        RelationshipType.UNKNOWN
    )

    custom_type: str | None = None

    attributes: dict[str, object] = field(
        default_factory=dict
    )

    trust: float = 0.0

    familiarity: float = 0.0

    emotional_bond: float = 0.0

    provenance: list[Provenance] = field(
        default_factory=list
    )

    history: list[UUID] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:

        self.trust = self._clamp(
            self.trust
        )

        self.familiarity = self._clamp(
            self.familiarity
        )

        self.emotional_bond = self._clamp(
            self.emotional_bond
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    def add_provenance(
        self,
        provenance: Provenance,
    ) -> None:

        self.provenance.append(
            provenance
        )

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
