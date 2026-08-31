from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from v_core.autonomy import AuthorizationGuard

from .access import EVMCapability
from .erc20 import ERC20Report, analyze_erc20_abi
from .findings import Finding
from .oracle import (
    OraclePolicy,
    OracleReport,
    OracleRound,
    SequencerStatus,
    validate_oracle_round,
)
from .security import analyze_solidity_security


@dataclass(slots=True)
class EVMToolkit:
    """Capability-gated deterministic EVM tools exposed to V."""

    authorization: AuthorizationGuard

    def analyze_erc20(self, abi: Iterable[dict[str, Any]]) -> ERC20Report:
        self.authorization.require(EVMCapability.ERC20_ANALYZE.value)
        return analyze_erc20_abi(abi)

    def validate_oracle(
        self,
        data: OracleRound,
        policy: OraclePolicy,
        *,
        now: int,
        sequencer: SequencerStatus | None = None,
    ) -> OracleReport:
        self.authorization.require(EVMCapability.ORACLE_VALIDATE.value)
        return validate_oracle_round(
            data,
            policy,
            now=now,
            sequencer=sequencer,
        )

    def analyze_security(self, source: str) -> tuple[Finding, ...]:
        self.authorization.require(EVMCapability.SECURITY_ANALYZE.value)
        return analyze_solidity_security(source)
