from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from v_core.autonomy import AuthorizationGuard

from .access import EVMCapability
from .erc20 import ERC20Report, analyze_erc20_abi
from .findings import Finding
from .flash_swap import (
    uniswap_v2_cross_token_repayment,
    uniswap_v2_same_token_repayment,
    uniswap_v3_flash_fee,
)
from .oracle import (
    OraclePolicy,
    OracleReport,
    OracleRound,
    SequencerStatus,
    validate_oracle_round,
)
from .security import analyze_solidity_security
from .uniswap import HookPermissions, decode_v4_hook_address


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

    def decode_hook(self, address: str) -> HookPermissions:
        self.authorization.require(EVMCapability.UNISWAP_HOOKS_SIMULATE.value)
        return decode_v4_hook_address(address)

    def quote_v2_same_token_flash(self, amount_out: int) -> int:
        self.authorization.require(EVMCapability.FLASH_SWAP_SIMULATE.value)
        return uniswap_v2_same_token_repayment(amount_out)

    def quote_v2_cross_token_flash(
        self,
        amount_out: int,
        reserve_in: int,
        reserve_out: int,
    ) -> int:
        self.authorization.require(EVMCapability.FLASH_SWAP_SIMULATE.value)
        return uniswap_v2_cross_token_repayment(amount_out, reserve_in, reserve_out)

    def quote_v3_flash_fee(self, amount: int, fee_pips: int) -> int:
        self.authorization.require(EVMCapability.FLASH_SWAP_SIMULATE.value)
        return uniswap_v3_flash_fee(amount, fee_pips)
