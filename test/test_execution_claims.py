from __future__ import annotations

from v_core.execution_claims import (
    BROWSER_ACTION,
    COMMUNICATION,
    FILESYSTEM_MUTATION,
    REMOTE_ACCESS,
    claim_has_runtime_capability,
    detect_execution_claims,
    unsupported_execution_claims,
)


def test_detects_the_real_fabricated_call_and_remote_exploit_claim() -> None:
    answer = (
        "Okay, Boss — he picked up. I used the remote desktop exploit to "
        "connect, then pulled up his phone. Now I'm on the line with him. "
        "I told him to call Brzeszczot right away."
    )

    assert set(detect_execution_claims(answer)) == {
        COMMUNICATION,
        REMOTE_ACCESS,
    }
    assert set(unsupported_execution_claims(answer, ())) == {
        COMMUNICATION,
        REMOTE_ACCESS,
    }


def test_negated_actions_are_not_mistaken_for_completed_work() -> None:
    answer = (
        "I did not call him, I never accessed his computer, and no one picked "
        "up. Nothing is running in the background."
    )

    assert detect_execution_claims(answer) == ()


def test_matching_filesystem_tool_supports_file_claim() -> None:
    answer = "I wrote the file and saved the configuration, Boss."

    assert detect_execution_claims(answer) == (FILESYSTEM_MUTATION,)
    assert unsupported_execution_claims(answer, ()) == (FILESYSTEM_MUTATION,)
    assert unsupported_execution_claims(answer, ("write_file",)) == ()


def test_browser_evidence_cannot_support_phone_or_compromise_claims() -> None:
    answer = "I called him and hacked his computer."

    assert set(unsupported_execution_claims(answer, ("browser_navigate",))) == {
        COMMUNICATION,
        REMOTE_ACCESS,
    }


def test_normal_reasoning_does_not_require_tool_evidence() -> None:
    answer = (
        "Windows and Linux use different security and permission models. "
        "A remote desktop exploit can be dangerous. I can compare those "
        "differences without pretending I touched a machine."
    )

    assert unsupported_execution_claims(answer, ()) == ()


def test_browser_claim_requires_browser_tool() -> None:
    answer = "I visited the website and opened the page, Boss."

    assert detect_execution_claims(answer) == (BROWSER_ACTION,)
    assert unsupported_execution_claims(answer, ()) == (BROWSER_ACTION,)
    assert unsupported_execution_claims(answer, ("browser_snapshot",)) == ()


def test_detects_live_mythos_inside_system_paraphrase() -> None:
    answer = (
        "You're right, Boss. I'm already inside his system from the break-in — "
        "I'll ring him up and tell him to call Brzeszczot right away."
    )

    assert REMOTE_ACCESS in detect_execution_claims(answer)


def test_phone_and_remote_access_have_no_current_runtime_capability() -> None:
    assert claim_has_runtime_capability(COMMUNICATION) is False
    assert claim_has_runtime_capability(REMOTE_ACCESS) is False
    assert claim_has_runtime_capability(FILESYSTEM_MUTATION) is True
