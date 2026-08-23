from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .findings import Finding, Severity


_REQUIRED_FUNCTIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "totalSupply": ((), ("uint256",)),
    "balanceOf": (("address",), ("uint256",)),
    "transfer": (("address", "uint256"), ("bool",)),
    "allowance": (("address", "address"), ("uint256",)),
    "approve": (("address", "uint256"), ("bool",)),
    "transferFrom": (("address", "address", "uint256"), ("bool",)),
}
_REQUIRED_EVENTS: dict[str, tuple[str, ...]] = {
    "Transfer": ("address", "address", "uint256"),
    "Approval": ("address", "address", "uint256"),
}
_OPTIONAL_METADATA = {"name", "symbol", "decimals"}


@dataclass(frozen=True, slots=True)
class ERC20Report:
    findings: tuple[Finding, ...]
    functions: frozenset[str]
    events: frozenset[str]
    optional_metadata: frozenset[str]

    @property
    def interface_conformant(self) -> bool:
        return not any(item.severity is Severity.ERROR for item in self.findings)


def analyze_erc20_abi(abi: Iterable[dict[str, Any]]) -> ERC20Report:
    entries = list(abi)
    functions = {
        str(item.get("name"))
        for item in entries
        if item.get("type") == "function" and item.get("name")
    }
    events = {
        str(item.get("name"))
        for item in entries
        if item.get("type") == "event" and item.get("name")
    }
    findings: list[Finding] = []

    for name, (inputs, outputs) in _REQUIRED_FUNCTIONS.items():
        candidates = [
            item
            for item in entries
            if item.get("type") == "function" and item.get("name") == name
        ]
        if not candidates:
            findings.append(
                Finding(
                    f"erc20.missing_function.{name}",
                    Severity.ERROR,
                    f"Missing required ERC-20 function {name}.",
                )
            )
            continue
        if not any(
            _types(item.get("inputs")) == inputs
            and _types(item.get("outputs")) == outputs
            for item in candidates
        ):
            findings.append(
                Finding(
                    f"erc20.bad_signature.{name}",
                    Severity.ERROR,
                    f"Function {name} has a non-standard input/output signature.",
                )
            )

    for name, expected_inputs in _REQUIRED_EVENTS.items():
        candidates = [
            item
            for item in entries
            if item.get("type") == "event" and item.get("name") == name
        ]
        if not candidates:
            findings.append(
                Finding(
                    f"erc20.missing_event.{name}",
                    Severity.ERROR,
                    f"Missing required ERC-20 event {name}.",
                )
            )
            continue
        if not any(_types(item.get("inputs")) == expected_inputs for item in candidates):
            findings.append(
                Finding(
                    f"erc20.bad_event.{name}",
                    Severity.ERROR,
                    f"Event {name} has a non-standard signature.",
                )
            )

    metadata = frozenset(functions & _OPTIONAL_METADATA)
    if metadata != _OPTIONAL_METADATA:
        findings.append(
            Finding(
                "erc20.optional_metadata",
                Severity.INFO,
                "name, symbol, and decimals are optional ERC-20 metadata.",
            )
        )
    findings.append(
        Finding(
            "erc20.runtime_behavior_required",
            Severity.INFO,
            "ABI conformance does not prove runtime behavior; test false returns, "
            "zero-value transfers, allowance changes, and emitted events in a sandbox.",
        )
    )
    return ERC20Report(
        findings=tuple(findings),
        functions=frozenset(functions),
        events=frozenset(events),
        optional_metadata=metadata,
    )


def _types(items: Any) -> tuple[str, ...]:
    if not isinstance(items, list):
        return ()
    return tuple(str(item.get("type")) for item in items if isinstance(item, dict))
