"""
V-Core fundamental domain models.

This package contains persona-agnostic primitives used by the
runtime, cognition, memory, relationships, and learning layers.
"""

from .entities import Entity, EntityType
from .relationships import Relationship, RelationshipType
from .provenance import Provenance, ProvenanceSource
from .observations import Observation, ObservationType
from .state import CoreState

__all__ = [
    "Entity",
    "EntityType",
    "Relationship",
    "RelationshipType",
    "Provenance",
    "ProvenanceSource",
    "Observation",
    "ObservationType",
    "CoreState",
]
