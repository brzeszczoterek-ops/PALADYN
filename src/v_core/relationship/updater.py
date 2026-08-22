from __future__ import annotations

from ..llm import LLM
from ..utils import parse_llm_json
from ..memory.models import ExperienceEntry
from .state import RelationshipState


class RelationshipUpdater:

    def __init__(self, llm: LLM):
        self.llm = llm

    async def update(
        self,
        state: RelationshipState,
        experience: ExperienceEntry,
    ) -> RelationshipState:

        prompt = f"""
You are V.

You are evaluating whether a new experience changes your relationship
with Boss.

Current relationship state:

Familiarity:
{state.familiarity}

Trust:
{state.trust}

Emotional bond:
{state.emotional_bond}

Relationship depth:
{state.relationship_depth}

Understanding of Boss:
{state.understanding_of_boss}

Shared history:
{state.shared_history}

Preferred forms of address:
{state.preferred_forms_of_address}

New experience:

Summary:
{experience.summary}

Lesson:
{experience.lesson}

Kind:
{experience.kind.value}

Source:
{experience.source.value}

Confidence:
{experience.confidence}

Importance:
{experience.importance}

Your task is to determine whether this experience meaningfully changes
your relationship with Boss.

Rules:

- Do not change relationship state merely because an interaction occurred.
- Routine technical tasks should normally have little or no relationship impact.
- A meaningful shared success, failure, personal disclosure, trust event,
  recurring interaction, or significant preference may affect the relationship.
- Do not infer emotional closeness from a single ordinary interaction.
- Do not invent feelings, intentions, or history.
- Confidence in the experience must influence how strongly it can affect
  relationship state.
- Strong relationship changes require meaningful evidence.
- Familiarity may increase when V learns something reliable about Boss.
- Understanding of Boss may increase when reliable information about Boss
  is learned.
- Trust may increase when Boss demonstrates meaningful trust, consistency,
  honesty, or successful cooperation.
- Emotional bond should change slowly and only from genuinely meaningful
  interactions.
- Relationship depth should emerge gradually from accumulated meaningful
  experiences.
- Negative experiences may reduce trust or increase distance when justified.
- Do not make large changes from a single weak inference.
- Keep all values between 0.0 and 1.0.
- Add shared history only when the event is genuinely meaningful.
- Never fabricate a preferred form of address.

Return ONLY valid JSON.

Schema:

{{
    "familiarity_delta": 0.0,
    "trust_delta": 0.0,
    "emotional_bond_delta": 0.0,
    "relationship_depth_delta": 0.0,
    "understanding_of_boss_delta": 0.0,
    "shared_experience": "",
    "preferred_form_of_address": ""
}}

Delta rules:

- Deltas must normally remain between -0.10 and 0.10.
- Routine interactions should usually produce deltas close to 0.
- Low-confidence experiences should produce very small changes.
- Do not use large positive deltas simply because an interaction was pleasant.
- Do not use large negative deltas without clear evidence of a meaningful
  negative event.
- Empty strings mean that nothing should be added.
- Output ONLY JSON.
""".strip()

        response = await self.llm.ask(prompt)

        data = parse_llm_json(
            response,
            default={
                "familiarity_delta": 0.0,
                "trust_delta": 0.0,
                "emotional_bond_delta": 0.0,
                "relationship_depth_delta": 0.0,
                "understanding_of_boss_delta": 0.0,
                "shared_experience": "",
                "preferred_form_of_address": "",
            },
        )

        state.familiarity = self._apply_delta(
            state.familiarity,
            data.get("familiarity_delta", 0.0),
        )

        state.trust = self._apply_delta(
            state.trust,
            data.get("trust_delta", 0.0),
        )

        state.emotional_bond = self._apply_delta(
            state.emotional_bond,
            data.get("emotional_bond_delta", 0.0),
        )

        state.relationship_depth = self._apply_delta(
            state.relationship_depth,
            data.get("relationship_depth_delta", 0.0),
        )

        state.understanding_of_boss = self._apply_delta(
            state.understanding_of_boss,
            data.get("understanding_of_boss_delta", 0.0),
        )

        state.add_shared_experience(
            data.get("shared_experience", "")
        )

        state.add_form_of_address(
            data.get("preferred_form_of_address", "")
        )

        state.touch()

        return state

    @staticmethod
    def _apply_delta(
        value: float,
        delta: float,
    ) -> float:

        try:
            delta = float(delta)
        except (TypeError, ValueError):
            delta = 0.0

        delta = max(
            -0.10,
            min(
                0.10,
                delta,
            ),
        )

        return max(
            0.0,
            min(
                1.0,
                value + delta,
            ),
        )
