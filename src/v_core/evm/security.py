from __future__ import annotations

import re

from .findings import Finding, Severity


def analyze_solidity_security(source: str) -> tuple[Finding, ...]:
    """Conservative source lint for wrapper and boundary mistakes.

    This intentionally does not claim to be a Solidity parser or an audit. It
    catches high-signal omissions before compilation, property tests, fuzzing,
    invariant tests, and human review.
    """

    findings: list[Finding] = []
    compact = re.sub(r"//.*?$|/\*.*?\*/", "", source, flags=re.M | re.S)

    token_calls = re.search(r"\.\s*(transfer|transferFrom|approve)\s*\(", compact)
    if token_calls and "SafeERC20" not in compact:
        findings.append(
            Finding(
                "security.unsafe_erc20_call",
                Severity.WARNING,
                "ERC-20 calls are present without an observable SafeERC20 wrapper.",
            )
        )
    if "tx.origin" in compact:
        findings.append(
            Finding(
                "security.tx_origin",
                Severity.ERROR,
                "tx.origin must not be used for authorization.",
            )
        )
    if ".delegatecall(" in compact:
        findings.append(
            Finding(
                "security.delegatecall",
                Severity.WARNING,
                "delegatecall crosses a storage and code trust boundary; review manually.",
            )
        )
    if "selfdestruct(" in compact:
        findings.append(
            Finding(
                "security.selfdestruct",
                Severity.WARNING,
                "selfdestruct semantics are chain/fork dependent and require explicit review.",
            )
        )
    if ".call{" in compact and "ReentrancyGuard" not in compact:
        findings.append(
            Finding(
                "security.external_value_call",
                Severity.WARNING,
                "External value call found without an observable ReentrancyGuard.",
            )
        )
    if "latestRoundData(" in compact:
        if "updatedAt" not in compact:
            findings.append(
                Finding(
                    "security.oracle_freshness",
                    Severity.ERROR,
                    "latestRoundData is used without an observable updatedAt freshness check.",
                )
            )
        if not re.search(r"answer\s*(>|>=|!=)\s*0", compact):
            findings.append(
                Finding(
                    "security.oracle_answer",
                    Severity.WARNING,
                    "Oracle answer has no observable positive-value check.",
                )
            )
    if re.search(r"function\s+(pause|unpause)\s*\([^)]*\)\s*(external|public)", compact):
        if not re.search(r"function\s+(pause|unpause).*?(onlyOwner|onlyRole|restricted)", compact, re.S):
            findings.append(
                Finding(
                    "security.unprotected_pause",
                    Severity.ERROR,
                    "Public pause control has no observable access-control modifier.",
                )
            )

    findings.append(
        Finding(
            "security.scope",
            Severity.INFO,
            "Heuristic lint is not an audit; compile and run unit, fuzz, invariant, and fork tests.",
        )
    )
    return tuple(findings)
