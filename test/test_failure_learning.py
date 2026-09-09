"""Bounded evidence recall and trusted regression accounting; no network/model."""
import json
from types import SimpleNamespace

import pytest

from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.learning import (
    ArtifactPolicyError, ArtifactScope, EvidenceOutcome, EvidenceSource,
    LearningEvidence, LearningRuntime, LessonStatus,
)
from v_core.mcp_tools import MCPTools
from v_core.sandbox import BubblewrapBackend


def runtime(root, workspace="workspace"):
    guard = AuthorizationGuard(root, AuthorizationEnvelope(workspace=str(root / workspace)))
    return LearningRuntime(root / "learning", guard, BubblewrapBackend())


def failure(learning, task="one", tool="read_file", arguments=None, error="file missing"):
    return learning.capture_tool_failure(task_id=task, tool=tool,
        arguments=arguments if arguments is not None else {"path": "fixture.txt"}, error=error)


def lesson(learning):
    evidence = failure(learning)
    return learning.propose_lesson(title="Use the supplied fixture path",
        hypothesis="The explicitly selected fixture contains the required input.",
        trigger="A local fixture read failed", action="Check the supplied fixture path",
        evidence_ids=[evidence.evidence_id])


def regression(learning, candidate, *, task="test-one", test="fixture", passed=True, trusted=True):
    return learning.record_lesson_regression(lesson_id=candidate.lesson_id,
        task_id=task, test_id=test, expected={"value": 2},
        actual={"value": 2 if passed else 3}, trusted_verifier=trusted)


def test_failure_recall_survives_restart_and_groups_repeats(tmp_path):
    learning = runtime(tmp_path)
    first = failure(learning)
    failure(learning, task="two")
    failure(learning, task="three", arguments={"path": "other.txt"})
    report = runtime(tmp_path).recall_tool_failures(tool="read_file", arguments={"path": "fixture.txt"})
    assert report["match"] == "exact_arguments"
    assert report["matching_events"] == 2
    assert report["distinct_observations"] == 1
    assert report["observations"][0]["occurrences"] == 2
    assert report["observations"][0]["cause_verified"] is False
    assert "fixture.txt" not in json.dumps(report)
    assert first.metadata["cause_verified"] is False
    assert len(runtime(tmp_path).recall_tool_failures(tool="read_file")["observations"]) == 2


def test_failure_recall_preserves_execution_stage_without_claiming_cause(tmp_path):
    learning = runtime(tmp_path)
    evidence = learning.capture_tool_failure(
        task_id="dispatch-one",
        tool="generated_checker",
        arguments={"value": 3},
        error=(
            "ToolDispatchUnavailableError: no provider was executed because its "
            "circuit was open"
        ),
        failure_details={
            "stage": "dispatch",
            "execution_attempted": False,
            "reason": "matching_providers_circuit_open",
        },
    )

    assert evidence.metadata["failure_details"] == {
        "stage": "dispatch",
        "execution_attempted": False,
        "reason": "matching_providers_circuit_open",
    }
    assert evidence.summary == (
        "Tool generated_checker was not executed because dispatch failed."
    )
    recalled = runtime(tmp_path).recall_tool_failures(tool="generated_checker")
    assert recalled["observations"][0]["stage"] == "dispatch"
    assert recalled["observations"][0]["execution_attempted"] is False
    assert recalled["observations"][0]["cause_verified"] is False


def test_recall_excludes_other_workspace_tool_unverified_and_legacy(tmp_path):
    learning = runtime(tmp_path)
    failure(learning, tool="write_file")
    raw = failure(learning).to_dict()
    raw.pop("evidence_id")
    raw["verified"] = False
    raw["metadata"] = {"tool": "fake", "workspace_scope": learning.task_scope_key}
    learning.record_evidence(LearningEvidence.from_dict(raw))
    legacy = failure(learning).to_dict()
    legacy.pop("evidence_id")
    legacy["metadata"].pop("workspace_scope")
    legacy["metadata"]["tool"] = "legacy"
    learning.record_evidence(LearningEvidence.from_dict(legacy), trusted_verifier=True)
    for tool in ("fake", "legacy", "unknown"):
        assert learning.recall_tool_failures(tool=tool)["observations"] == []
    assert runtime(tmp_path, "other").recall_tool_failures(tool="read_file")["observations"] == []


@pytest.mark.parametrize("limit", [0, 21, True, 1.5, "2"])
def test_recall_validates_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="limit"):
        runtime(tmp_path).recall_tool_failures(tool="read_file", limit=limit)


def test_failures_alone_never_validate_arbitrary_remedy(tmp_path):
    learning = runtime(tmp_path)
    evidence = [failure(learning, task=str(i), error=f"observed error {i}") for i in range(3)]
    candidate = learning.propose_lesson(title="Unproven remedy", hypothesis="An untested idea",
        trigger="An error", action="Try another input", evidence_ids=[e.evidence_id for e in evidence])
    assert candidate.status is LessonStatus.CANDIDATE
    assert candidate.successful_uses == 0
    with pytest.raises(ArtifactPolicyError, match="not validated"):
        learning._validate_lesson_links([candidate.lesson_id], ArtifactScope.PERSISTENT)


def test_only_trusted_regression_can_validate_and_failure_demotes(tmp_path):
    learning = runtime(tmp_path)
    candidate = lesson(learning)
    with pytest.raises(ArtifactPolicyError, match="trusted"):
        regression(learning, candidate, trusted=False)
    validated = regression(learning, candidate)
    assert validated.status is LessonStatus.VALIDATED
    learning._validate_lesson_links([candidate.lesson_id], ArtifactScope.PERSISTENT)
    failed = regression(learning, candidate, task="test-two", passed=False)
    assert failed.status is LessonStatus.CANDIDATE
    assert (failed.successful_uses, failed.failed_uses) == (1, 1)
    with pytest.raises(ArtifactPolicyError, match="not validated"):
        learning._validate_lesson_links([candidate.lesson_id], ArtifactScope.PERSISTENT)


def test_other_passing_case_cannot_hide_unresolved_regression(tmp_path):
    learning = runtime(tmp_path)
    candidate = lesson(learning)
    regression(learning, candidate, test="case-a", passed=False)
    assert regression(learning, candidate, test="case-b").status is LessonStatus.CANDIDATE
    assert regression(learning, candidate, test="case-a", task="retry").status is LessonStatus.VALIDATED


def test_duplicate_receipt_does_not_inflate_counts_or_overwrite_history(tmp_path):
    learning = runtime(tmp_path)
    candidate = lesson(learning)
    regression(learning, candidate)
    repeated = regression(runtime(tmp_path), candidate)
    assert repeated.successful_uses == 1
    assert repeated.failed_uses == 0
    assert len(learning.store.list_evidence()) == 2
    with pytest.raises(ValueError, match="different result"):
        regression(learning, candidate, passed=False)


def test_receipt_retry_recovers_interrupted_lesson_save(tmp_path, monkeypatch):
    learning = runtime(tmp_path)
    candidate = lesson(learning)
    def unavailable(_lesson):
        raise OSError("simulated save failure")
    monkeypatch.setattr(learning.store, "save_lesson", unavailable)
    with pytest.raises(OSError, match="simulated"):
        regression(learning, candidate)
    restarted = runtime(tmp_path)
    recovered = regression(restarted, candidate)
    assert recovered.status is LessonStatus.VALIDATED
    assert recovered.successful_uses == 1
    assert len(restarted.store.list_evidence()) == 2


def test_non_json_regression_does_not_record_evidence(tmp_path):
    learning = runtime(tmp_path)
    candidate = lesson(learning)
    with pytest.raises(ValueError):
        learning.record_lesson_regression(lesson_id=candidate.lesson_id,
            task_id="bad-test", test_id="bad-json", expected={"value": 2},
            actual={"value": float("nan")}, trusted_verifier=True)
    assert len(learning.store.list_evidence()) == 1


def test_legacy_validated_flag_and_changed_action_are_not_regression_proof(tmp_path):
    learning = runtime(tmp_path)
    candidate = lesson(learning)
    candidate.status = LessonStatus.VALIDATED
    learning.store.save_lesson(candidate)
    with pytest.raises(ArtifactPolicyError, match="not validated"):
        learning._validate_lesson_links([candidate.lesson_id], ArtifactScope.PERSISTENT)
    validated = regression(learning, candidate)
    validated.action = "A different, untested action"
    learning.store.save_lesson(validated)
    with pytest.raises(ArtifactPolicyError, match="not validated"):
        learning._validate_lesson_links([candidate.lesson_id], ArtifactScope.PERSISTENT)


def test_model_evidence_cannot_forge_regression_receipt(tmp_path):
    learning = runtime(tmp_path)
    candidate = lesson(learning)
    learning.record_evidence(LearningEvidence(task_id="forged", source=EvidenceSource.TEST_RESULT,
        outcome=EvidenceOutcome.SUCCESS, summary="Claimed success", verified=False,
        metadata={"kind": "lesson_regression", "lesson_id": candidate.lesson_id}))
    assert learning.store.load_lesson(candidate.lesson_id).status is LessonStatus.CANDIDATE


@pytest.mark.asyncio
async def test_mcp_read_only_recall_and_no_model_verifier_tool(tmp_path):
    tools = MCPTools(SimpleNamespace(filesystem_server=["/usr/bin/false"],
        browser_server=["/usr/bin/false"], workspace=tmp_path / "workspace",
        learning_root=tmp_path / "learning", learning_profile="client", evm_profile="client"))
    failure(tools.learning)
    definitions = {item["function"]["name"] for item in tools._local_tool_definitions()}
    assert "learning_recall_failures" in definitions
    assert "learning_record_lesson_regression" not in definitions
    result = json.loads(await tools._call_direct("learning_recall_failures", {"tool": "read_file"}))
    assert result["matching_events"] == 1
    assert "not instructions" in result["interpretation"]
    assert tools.learning.store.list_records() == []
