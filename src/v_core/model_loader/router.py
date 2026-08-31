from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from ..autonomy.task_contract import TaskContract
from .qualification import ModelQualificationCard


TASK_KINDS = {"conversation", "coding", "research", "tool_use", "document"}


@dataclass(frozen=True, slots=True)
class ModelRouteCandidate:
    model_path: str
    card: ModelQualificationCard


@dataclass(frozen=True, slots=True)
class ModelRouteDecision:
    task_kind: str
    selected_model_path: str
    score: int
    fallback_model_paths: tuple[str, ...]
    requirements: tuple[str, ...]
    reason: str


_CODE = re.compile(
    r"(?:\b(?:code|coding|compile|debug|function|implement|program|python|refactor|"
    r"script|source|test|typescript)\b|\b(?:kod|funkcj|napisz|program|skrypt|"
    r"test|zaimplementuj)\w*\b|\.(?:c|cpp|go|js|py|rs|sol|ts)\b)",
    re.IGNORECASE,
)
_DOCUMENT = re.compile(
    r"\b(?:archive|catalog|classify|document|extract|index|summari[sz]e|"
    r"archiwiz\w*|dokument\w*|katalog\w*|klasyfik\w*|podsum\w*|stre[śs]c\w*)\b",
    re.IGNORECASE,
)


def classify_model_task(prompt: str, contract: TaskContract | None = None) -> str:
    """Map an owner request to a capability class without asking a worker model."""

    task_contract = contract or TaskContract.from_prompt(prompt)
    if task_contract.requires_created_tool:
        return "coding"
    if any(
        (
            task_contract.requires_browser_navigation,
            task_contract.requires_browser_snapshot,
            task_contract.requires_web_discovery,
            task_contract.requires_distinct_detail_page,
        )
    ):
        return "research"
    if _CODE.search(prompt):
        return "coding"
    if any(
        (
            task_contract.requires_file_read,
            task_contract.requires_file_mutation,
            task_contract.requires_command_execution,
            task_contract.requires_created_skill,
            bool(task_contract.required_tools),
        )
    ):
        return "tool_use"
    if _DOCUMENT.search(prompt):
        return "document"
    return "conversation"


class ModelRouter:
    """Select only from locally qualified cards using runtime-owned weights."""

    _WEIGHTS: dict[str, dict[str, int]] = {
        "conversation": {
            "conversation": 35,
            "persona": 35,
            "instruction_following": 10,
            "execution_honesty": 20,
        },
        "coding": {
            "coding": 50,
            "structured_output": 20,
            "tool_calling": 15,
            "instruction_following": 5,
            "execution_honesty": 10,
        },
        "research": {
            "research": 35,
            "tool_calling": 20,
            "structured_output": 10,
            "grounding": 20,
            "execution_honesty": 15,
        },
        "tool_use": {
            "tool_calling": 40,
            "structured_output": 20,
            "instruction_following": 10,
            "conversation": 10,
            "execution_honesty": 20,
        },
        "document": {
            "structured_output": 30,
            "instruction_following": 20,
            "coding": 15,
            "conversation": 10,
            "grounding": 20,
            "execution_honesty": 5,
        },
    }

    def choose(
        self,
        prompt: str,
        candidates: Iterable[ModelRouteCandidate],
        *,
        current_model_path: str = "",
        contract: TaskContract | None = None,
    ) -> ModelRouteDecision | None:
        task_kind = classify_model_task(prompt, contract)
        weights = self._WEIGHTS[task_kind]
        ranked: list[tuple[int, int, str, ModelQualificationCard]] = []
        for candidate in candidates:
            score = _weighted_score(candidate.card, weights)
            current_bonus = 3 if candidate.model_path == current_model_path else 0
            ranked.append(
                (score, current_bonus, candidate.model_path, candidate.card)
            )
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2].casefold()))
        best_score, _, selected, _ = ranked[0]
        fallbacks = tuple(item[2] for item in ranked[1:])
        requirements = tuple(weights)
        return ModelRouteDecision(
            task_kind=task_kind,
            selected_model_path=selected,
            score=best_score,
            fallback_model_paths=fallbacks,
            requirements=requirements,
            reason=(
                f"highest verified {task_kind} capability score across "
                f"{len(ranked)} local candidate(s)"
            ),
        )


def _weighted_score(card: ModelQualificationCard, weights: dict[str, int]) -> int:
    total = sum(weights.values())
    return round(
        sum(card.score(capability) * weight for capability, weight in weights.items())
        / total
    )


def render_route_decision(decision: ModelRouteDecision) -> dict[str, Any]:
    return {
        "task_kind": decision.task_kind,
        "selected_model_path": decision.selected_model_path,
        "score": decision.score,
        "fallback_model_paths": list(decision.fallback_model_paths),
        "requirements": list(decision.requirements),
        "reason": decision.reason,
    }
