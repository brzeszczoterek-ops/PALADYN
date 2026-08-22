from __future__ import annotations

from uuid import UUID

from v_core.core import (
    CoreState,
    Entity,
    EntityType,
    Observation,
    ObservationType,
    Provenance,
    ProvenanceSource,
    Relationship,
    RelationshipType,
)


def test_entity_can_be_created() -> None:

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Paweł",
    )

    assert isinstance(
        entity.entity_id,
        UUID,
    )

    assert entity.entity_type == (
        EntityType.PERSON
    )

    assert entity.name == "Paweł"


def test_entity_aliases_are_unique() -> None:

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Paweł",
    )

    entity.add_alias("Boss")
    entity.add_alias("Boss")
    entity.add_alias("")

    assert entity.aliases == [
        "Boss"
    ]


def test_entity_matches_name_or_alias() -> None:

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Paweł",
    )

    entity.add_alias("Boss")

    assert entity.matches_name(
        "Paweł"
    )

    assert entity.matches_name(
        "paweł"
    )

    assert entity.matches_name(
        "Boss"
    )

    assert entity.matches_name(
        "boss"
    )

    assert not entity.matches_name(
        "Alice"
    )


def test_entity_attributes_are_generic() -> None:

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Paweł",
    )

    entity.set_attribute(
        "preferred_name",
        "Boss",
    )

    assert entity.get_attribute(
        "preferred_name"
    ) == "Boss"

    assert entity.get_attribute(
        "does_not_exist"
    ) is None


def test_provenance_clamps_confidence() -> None:

    low = Provenance(
        source=ProvenanceSource.DIRECTLY_TOLD,
        confidence=-10,
    )

    high = Provenance(
        source=ProvenanceSource.VERIFIED,
        confidence=10,
    )

    assert low.confidence == 0.0
    assert high.confidence == 1.0


def test_provenance_records_origin() -> None:

    provenance = Provenance(
        source=ProvenanceSource.DIRECTLY_TOLD,
        confidence=1.0,
    )

    assert provenance.source == (
        ProvenanceSource.DIRECTLY_TOLD
    )

    assert provenance.confidence == 1.0


def test_relationship_is_independent_from_entities() -> None:

    person = Entity(
        entity_type=EntityType.PERSON,
        name="Paweł",
    )

    persona = Entity(
        entity_type=EntityType.PERSONA,
        name="V",
    )

    relationship = Relationship(
        source_entity_id=persona.entity_id,
        target_entity_id=person.entity_id,
        relationship_type=(
            RelationshipType.COLLABORATOR
        ),
    )

    assert (
        relationship.source_entity_id
        == persona.entity_id
    )

    assert (
        relationship.target_entity_id
        == person.entity_id
    )

    assert relationship.relationship_type == (
        RelationshipType.COLLABORATOR
    )


def test_relationship_metrics_are_clamped() -> None:

    relationship = Relationship(
        trust=2.0,
        familiarity=-1.0,
        emotional_bond=50.0,
    )

    assert relationship.trust == 1.0
    assert relationship.familiarity == 0.0
    assert relationship.emotional_bond == 1.0


def test_relationship_can_have_attributes() -> None:

    relationship = Relationship()

    relationship.set_attribute(
        "role",
        "Boss",
    )

    assert relationship.get_attribute(
        "role"
    ) == "Boss"


def test_observation_is_not_memory() -> None:

    observation = Observation(
        observation_type=(
            ObservationType.STATEMENT
        ),
        content="I am Boss.",
    )

    assert observation.content == (
        "I am Boss."
    )

    assert observation.observation_type == (
        ObservationType.STATEMENT
    )

    assert observation.is_empty() is False


def test_empty_observation_is_detected() -> None:

    observation = Observation(
        observation_type=(
            ObservationType.STATEMENT
        ),
        content="   ",
    )

    assert observation.is_empty() is True


def test_observation_can_have_provenance() -> None:

    provenance = Provenance(
        source=ProvenanceSource.DIRECTLY_TOLD,
        confidence=1.0,
    )

    observation = Observation(
        observation_type=(
            ObservationType.STATEMENT
        ),
        content="My preferred name is Boss.",
        provenance=provenance,
    )

    assert observation.provenance is provenance

    assert observation.provenance.source == (
        ProvenanceSource.DIRECTLY_TOLD
    )


def test_core_state_stores_entities() -> None:

    state = CoreState()

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Paweł",
    )

    state.add_entity(
        entity
    )

    assert state.get_entity(
        entity.entity_id
    ) is entity


def test_core_state_stores_relationships() -> None:

    state = CoreState()

    relationship = Relationship(
        relationship_type=(
            RelationshipType.FRIEND
        ),
    )

    state.add_relationship(
        relationship
    )

    assert state.get_relationship(
        relationship.relationship_id
    ) is relationship


def test_core_state_stores_observations() -> None:

    state = CoreState()

    observation = Observation(
        observation_type=(
            ObservationType.STATEMENT
        ),
        content="I am Boss.",
    )

    state.add_observation(
        observation
    )

    assert state.observations == [
        observation
    ]


def test_core_state_updates_timestamp() -> None:

    state = CoreState()

    original_timestamp = (
        state.updated_at
    )

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Paweł",
    )

    state.add_entity(
        entity
    )

    assert state.updated_at >= (
        original_timestamp
    )


def test_core_state_is_persona_agnostic() -> None:

    state = CoreState()

    v = Entity(
        entity_type=EntityType.PERSONA,
        name="V",
    )

    another_persona = Entity(
        entity_type=EntityType.PERSONA,
        name="AnotherPersona",
    )

    user = Entity(
        entity_type=EntityType.PERSON,
        name="Paweł",
    )

    state.add_entity(v)
    state.add_entity(another_persona)
    state.add_entity(user)

    v_relationship = Relationship(
        source_entity_id=v.entity_id,
        target_entity_id=user.entity_id,
        relationship_type=(
            RelationshipType.COLLABORATOR
        ),
    )

    another_relationship = Relationship(
        source_entity_id=(
            another_persona.entity_id
        ),
        target_entity_id=user.entity_id,
        relationship_type=(
            RelationshipType.USER
        ),
    )

    state.add_relationship(
        v_relationship
    )

    state.add_relationship(
        another_relationship
    )

    assert len(state.entities) == 3
    assert len(state.relationships) == 2

    assert (
        v_relationship.target_entity_id
        == another_relationship.target_entity_id
    )


def test_same_entity_can_have_different_relationships() -> None:

    state = CoreState()

    persona_a = Entity(
        entity_type=EntityType.PERSONA,
        name="V",
    )

    persona_b = Entity(
        entity_type=EntityType.PERSONA,
        name="Alice",
    )

    person = Entity(
        entity_type=EntityType.PERSON,
        name="Paweł",
    )

    state.add_entity(persona_a)
    state.add_entity(persona_b)
    state.add_entity(person)

    relationship_a = Relationship(
        source_entity_id=persona_a.entity_id,
        target_entity_id=person.entity_id,
        relationship_type=(
            RelationshipType.COLLABORATOR
        ),
    )

    relationship_b = Relationship(
        source_entity_id=persona_b.entity_id,
        target_entity_id=person.entity_id,
        relationship_type=(
            RelationshipType.FRIEND
        ),
    )

    state.add_relationship(
        relationship_a
    )

    state.add_relationship(
        relationship_b
    )

    assert relationship_a.relationship_type == (
        RelationshipType.COLLABORATOR
    )

    assert relationship_b.relationship_type == (
        RelationshipType.FRIEND
    )

    assert (
        relationship_a.target_entity_id
        == relationship_b.target_entity_id
    )
