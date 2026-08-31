from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.autonomy.policy import AuthorizationDenied
from v_core.evm import (
    EVMAccessProfile,
    EVMToolkit,
    OraclePolicy,
    OracleRound,
    SequencerStatus,
    analyze_erc20_abi,
    analyze_solidity_security,
)


def _erc20_abi() -> list[dict[str, object]]:
    def function(name: str, inputs: list[str], outputs: list[str]) -> dict[str, object]:
        return {
            "type": "function",
            "name": name,
            "inputs": [{"type": item} for item in inputs],
            "outputs": [{"type": item} for item in outputs],
        }

    def event(name: str, inputs: list[str]) -> dict[str, object]:
        return {
            "type": "event",
            "name": name,
            "inputs": [{"type": item} for item in inputs],
        }

    return [
        function("totalSupply", [], ["uint256"]),
        function("balanceOf", ["address"], ["uint256"]),
        function("transfer", ["address", "uint256"], ["bool"]),
        function("allowance", ["address", "address"], ["uint256"]),
        function("approve", ["address", "uint256"], ["bool"]),
        function("transferFrom", ["address", "address", "uint256"], ["bool"]),
        event("Transfer", ["address", "address", "uint256"]),
        event("Approval", ["address", "address", "uint256"]),
    ]


def test_public_erc20_analysis_checks_exact_standard_interface() -> None:
    report = analyze_erc20_abi(_erc20_abi())
    broken = analyze_erc20_abi(_erc20_abi()[:-1])

    assert report.interface_conformant
    assert not broken.interface_conformant


def test_public_oracle_validation_is_capability_gated(tmp_path: Path) -> None:
    envelope = EVMAccessProfile.client().apply(
        AuthorizationEnvelope(workspace=str(tmp_path))
    )
    toolkit = EVMToolkit(AuthorizationGuard(tmp_path, envelope))
    report = toolkit.validate_oracle(
        OracleRound(10, 2_500_00000000, 800, 900, 10, 8),
        OraclePolicy(max_age_seconds=30, minimum_answer=Decimal("1")),
        now=1_000,
        sequencer=SequencerStatus(is_up=True, started_at=950),
    )

    assert report.value == Decimal("2500")
    assert not report.acceptable


def test_public_security_lint_flags_high_signal_errors(tmp_path: Path) -> None:
    envelope = EVMAccessProfile.client().apply(
        AuthorizationEnvelope(workspace=str(tmp_path))
    )
    toolkit = EVMToolkit(AuthorizationGuard(tmp_path, envelope))
    findings = toolkit.analyze_security(
        "require(tx.origin == owner); token.transferFrom(from, address(this), amount);"
    )

    assert {item.code for item in findings} >= {
        "security.tx_origin",
        "security.unsafe_erc20_call",
    }


def test_public_evm_toolkit_rejects_missing_capability(tmp_path: Path) -> None:
    toolkit = EVMToolkit(AuthorizationGuard(tmp_path, AuthorizationEnvelope()))

    with pytest.raises(AuthorizationDenied):
        toolkit.analyze_erc20(_erc20_abi())
