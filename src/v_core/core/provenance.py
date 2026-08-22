from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


class ProvenanceSource(str, Enum):
    DIRECTLY_TOLD = "directly_told"
    OBSERVED = "observed"
    INFERRED = "inferred"
    REFLECTED = "reflected"
    EXPERIENCED = "experienced"
    VERIFIED = "verified"
    IMPORTED = "imported"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Provenance:
    """
    Describes where a piece of information came from.

    Provenance is not a truth guarantee.
    It records origin and confidence so that later cognition,
    reflection, contradiction handling, and learning can reason
    about the information safely.
    """

    provenance_id: UUID = field(
        default_factory=uuid4
    )

    source: ProvenanceSource = (
        ProvenanceSource.UNKNOWN
    )

    confidence: float = 1.0

    source_entity_id: UUID | None = None

    source_reference: str | None = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.confidence = max(
            0.0,
            min(
                1.0,
                float(self.confidence),
            ),
        )
