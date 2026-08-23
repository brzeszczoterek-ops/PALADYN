from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .findings import Finding, Severity


@dataclass(frozen=True, slots=True)
class OracleRound:
    round_id: int
    answer: int
    started_at: int
    updated_at: int
    answered_in_round: int
    decimals: int


@dataclass(frozen=True, slots=True)
class SequencerStatus:
    is_up: bool
    started_at: int
    grace_period_seconds: int = 3_600


@dataclass(frozen=True, slots=True)
class OraclePolicy:
    max_age_seconds: int
    require_positive: bool = True
    minimum_answer: Decimal | None = None
    maximum_answer: Decimal | None = None

    def __post_init__(self) -> None:
        if self.max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        if (
            self.minimum_answer is not None
            and self.maximum_answer is not None
            and self.minimum_answer >= self.maximum_answer
        ):
            raise ValueError("minimum_answer must be lower than maximum_answer")


@dataclass(frozen=True, slots=True)
class OracleReport:
    value: Decimal | None
    age_seconds: int | None
    findings: tuple[Finding, ...]

    @property
    def acceptable(self) -> bool:
        return not any(item.severity is Severity.ERROR for item in self.findings)


def validate_oracle_round(
    data: OracleRound,
    policy: OraclePolicy,
    *,
    now: int,
    sequencer: SequencerStatus | None = None,
) -> OracleReport:
    findings: list[Finding] = []
    value: Decimal | None = None
    age: int | None = None

    if not 0 <= data.decimals <= 77:
        findings.append(
            Finding("oracle.decimals", Severity.ERROR, "Invalid oracle decimals.")
        )
    else:
        value = Decimal(data.answer) / (Decimal(10) ** data.decimals)

    if data.updated_at <= 0:
        findings.append(
            Finding("oracle.incomplete_round", Severity.ERROR, "Round has no update time.")
        )
    elif data.updated_at > now:
        findings.append(
            Finding("oracle.future_timestamp", Severity.ERROR, "Oracle update is in the future.")
        )
    else:
        age = now - data.updated_at
        if age > policy.max_age_seconds:
            findings.append(
                Finding(
                    "oracle.stale",
                    Severity.ERROR,
                    f"Oracle value is stale ({age}s > {policy.max_age_seconds}s).",
                )
            )

    if data.answered_in_round < data.round_id:
        findings.append(
            Finding(
                "oracle.round_mismatch",
                Severity.ERROR,
                "answeredInRound is older than roundId.",
            )
        )
    if policy.require_positive and data.answer <= 0:
        findings.append(
            Finding("oracle.non_positive", Severity.ERROR, "Oracle answer is not positive.")
        )
    if value is not None and policy.minimum_answer is not None:
        if value < policy.minimum_answer:
            findings.append(
                Finding("oracle.below_bound", Severity.ERROR, "Oracle answer is below policy bound.")
            )
    if value is not None and policy.maximum_answer is not None:
        if value > policy.maximum_answer:
            findings.append(
                Finding("oracle.above_bound", Severity.ERROR, "Oracle answer is above policy bound.")
            )

    if sequencer is not None:
        if not sequencer.is_up:
            findings.append(
                Finding("oracle.sequencer_down", Severity.ERROR, "L2 sequencer is down.")
            )
        elif now - sequencer.started_at <= sequencer.grace_period_seconds:
            findings.append(
                Finding(
                    "oracle.sequencer_grace_period",
                    Severity.ERROR,
                    "L2 sequencer recovery grace period has not elapsed.",
                )
            )

    return OracleReport(value=value, age_seconds=age, findings=tuple(findings))
