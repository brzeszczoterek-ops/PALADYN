from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from .provenance import Provenance


class ObservationType(str, Enum):
    STATEMENT = "statement"
    EVENT = "event"
    FACT = "fact"
    PREFERENCE = "preference"
    BEHAVIOUR = "behaviour"
    RELATIONSHIP_CLAIM = "relationship_claim"
    SELF_CLAIM = "self_claim"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Observation:
    """
    A piece of information perceived by the agent.

    Observation is deliberately NOT memory.

    An observation may later become:
        - a memory,
        - an inference,
        - a rejected claim,
        - a contradiction,
        - or simply disappear.

    This separation is essential for controlled learning.
    """

    observation_id: UUID = field(
        default_factory=uuid4
    )

    observation_type: ObservationType = (
        ObservationType.UNKNOWN
    )

    content: str = ""

    subject_entity_id: UUID | None = None

    object_entity_id: UUID | None = None

    attributes: dict[str, object] = field(
        default_factory=dict
    )

    provenance: Provenance | None = None

    observed_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    def is_empty(self) -> bool:

        return not bool(
            self.content.strip()
        )
