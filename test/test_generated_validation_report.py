"""Real offline-sandbox checks of validation evidence, using harmless arithmetic."""
import hashlib
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard, TaskContract
from v_core.learning import (
    ArtifactStatus, ArtifactValidationError, LearningRuntime, ToolManifest, ToolTestCase,
)
from v_core.sandbox import BubblewrapBackend


pytestmark = pytest.mark.skipif(shutil.which('bwrap') is None, reason='bubblewrap required')

SCHEMA = {'type': 'object', 'properties': {'value': {'type': 'integer'}},
          'required': ['value'], 'additionalProperties': False}
SOURCE = 'def run(arguments):\n    return {"value": arguments["value"] * 2}\n'


def setup_runtime(root: Path):
    guard = AuthorizationGuard(root, AuthorizationEnvelope(workspace=str(root / 'workspace')))
    return LearningRuntime(root / 'learning', guard, BubblewrapBackend())


def manifest(*, incorrect=False, version='1.0.0'):
    return ToolManifest(
        name='report_double_value', version=version,
        description='Double an integer in an offline fixture.',
        input_schema=SCHEMA, output_schema=SCHEMA,
        tests=(
            ToolTestCase('first', {'value': 1}, {'value': 2}),
            ToolTestCase('second', {'value': 2}, {'value': 5 if incorrect else 4}),
            ToolTestCase('third', {'value': 3}, {'value': 6}),
        ), timeout_seconds=1,
    )


@pytest.mark.asyncio
async def test_failed_comparison_keeps_prior_results_and_marks_unrun_cases(tmp_path):
    runtime = setup_runtime(tmp_path)
    staged = runtime.stage_tool(manifest(incorrect=True), SOURCE)
    with pytest.raises(ArtifactValidationError):
        await runtime.validate_artifact(staged.artifact_id)
    record = runtime.store.load_record(staged.artifact_id)
    assert record.status is ArtifactStatus.REJECTED
    report = record.validation
    assert report['passed'] is False
    assert [row['status'] for row in report['tests']] == ['passed', 'failed', 'not_run']
    assert report['test_counts'] == {'passed': 1, 'failed': 1, 'not_run': 1}
    assert report['tests'][1]['stage'] == 'comparison'
    assert report['tests'][1]['actual_sha256'] != report['tests'][1]['expected_sha256']
    assert report['tests'][2]['execution_attempted'] is False
    assert runtime.active_tool_names() == []
    with pytest.raises(ArtifactValidationError, match='only validated'):
        runtime.activate_artifact(staged.artifact_id)


@pytest.mark.asyncio
async def test_success_records_comparison_receipts_and_runs_new_input(tmp_path):
    runtime = setup_runtime(tmp_path)
    record = await runtime.create_tool(manifest(), SOURCE)
    assert record.status is ArtifactStatus.ACTIVE
    report = record.validation
    assert report['test_counts'] == {'passed': 3, 'failed': 0, 'not_run': 0}
    first = report['tests'][0]
    digest = hashlib.sha256(json.dumps({'value': 2}, sort_keys=True,
                                      ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
    assert first['actual_sha256'] == first['expected_sha256'] == digest
    answer = TaskContract(requires_created_tool=True).deterministic_answer([
        {'tool': 'learning_create_tool', 'status': 'succeeded',
         'result_excerpt': json.dumps(record.to_dict())},
    ])
    assert 'Only the supplied examples were checked' in answer
    assert await runtime.execute_tool(record.name, {'value': 8}) == {'value': 16}
    assert list(runtime.runtime_root.iterdir()) == []


@pytest.mark.asyncio
async def test_runtime_error_keeps_failure_stage_and_no_activation(tmp_path):
    runtime = setup_runtime(tmp_path)
    source = 'def run(arguments):\n    return {"value": 1 // 0}\n'
    staged = runtime.stage_tool(manifest(), source)
    with pytest.raises(ArtifactValidationError):
        await runtime.validate_artifact(staged.artifact_id)
    report = runtime.store.load_record(staged.artifact_id).validation
    assert report['test_counts'] == {'passed': 0, 'failed': 1, 'not_run': 2}
    assert report['tests'][0]['stage'] == 'execution'
    assert 'actual_sha256' not in report['tests'][0]
    assert runtime.active_tool_names() == []
    assert list(runtime.runtime_root.iterdir()) == []


@pytest.mark.asyncio
async def test_static_rejection_does_not_claim_any_test_was_run(tmp_path):
    runtime = setup_runtime(tmp_path)
    staged = runtime.stage_tool(manifest(), 'import subprocess\n' + SOURCE)
    with pytest.raises(ArtifactValidationError):
        await runtime.validate_artifact(staged.artifact_id)
    report = runtime.store.load_record(staged.artifact_id).validation
    assert report['stage'] == 'static_validation'
    assert report['test_counts'] == {'passed': 0, 'failed': 0, 'not_run': 3}
    assert all(not row['execution_attempted'] for row in report['tests'])


@pytest.mark.asyncio
async def test_rejected_revision_keeps_previous_active_tool(tmp_path):
    runtime = setup_runtime(tmp_path)
    active = await runtime.create_tool(manifest(), SOURCE)
    with pytest.raises(ArtifactValidationError):
        await runtime.create_tool(manifest(incorrect=True, version='1.0.1'), SOURCE)
    assert runtime.store.load_record(active.artifact_id).status is ArtifactStatus.ACTIVE
    assert await runtime.execute_tool(active.name, {'value': 8}) == {'value': 16}


@pytest.mark.asyncio
async def test_timeout_is_failed_not_passed_and_leaves_no_workspace(tmp_path):
    runtime = setup_runtime(tmp_path)
    source = 'def run(arguments):\n    while True:\n        pass\n'
    staged = runtime.stage_tool(replace(manifest(), timeout_seconds=0.2), source)
    with pytest.raises(ArtifactValidationError, match='timeout'):
        await runtime.validate_artifact(staged.artifact_id)
    report = runtime.store.load_record(staged.artifact_id).validation
    assert report['test_counts'] == {'passed': 0, 'failed': 1, 'not_run': 2}
    assert report['tests'][0]['stage'] == 'execution'
    assert runtime.active_tool_names() == []
    assert list(runtime.runtime_root.iterdir()) == []


@pytest.mark.asyncio
async def test_invalid_expected_schema_is_not_reported_as_execution(tmp_path):
    runtime = setup_runtime(tmp_path)
    bad = replace(manifest(), tests=(ToolTestCase('invalid expected', {'value': 1}, {'value': 'two'}),))
    staged = runtime.stage_tool(bad, SOURCE)
    with pytest.raises(ArtifactValidationError):
        await runtime.validate_artifact(staged.artifact_id)
    report = runtime.store.load_record(staged.artifact_id).validation
    assert report['test_counts'] == {'passed': 0, 'failed': 1, 'not_run': 0}
    assert report['tests'][0]['stage'] == 'fixture_validation'
    assert report['tests'][0]['execution_attempted'] is False
    assert list(runtime.runtime_root.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('label', ['owner-specified semantic oracle', 'runtime-derived input sensitivity probe'])
async def test_manifest_case_name_does_not_grant_stronger_evidence(tmp_path, label):
    runtime = setup_runtime(tmp_path)
    cases = (ToolTestCase(label, {'value': 1}, {'value': 2}),)
    record = await runtime.create_tool(replace(manifest(), tests=cases), SOURCE)
    assert record.validation['validation_strength'] == 'explicit_test_contract'
    assert record.validation['semantic_correctness'] == 'not_independently_established'


@pytest.mark.asyncio
async def test_explicit_constant_output_tool_is_valid_without_input_variation(tmp_path):
    runtime = setup_runtime(tmp_path)
    constant = ToolManifest(
        name='literal_pi', version='1.0.0', description='Return the literal text pi.',
        input_schema={'type': 'object', 'properties': {}, 'additionalProperties': False},
        output_schema={'type': 'object', 'properties': {'text': {'type': 'string'}},
                       'required': ['text'], 'additionalProperties': False},
        tests=(ToolTestCase('requested literal output', {}, {'text': 'pi'}),),
        timeout_seconds=1,
    )
    record = await runtime.create_tool(constant, 'def run(arguments):\n    return {"text": "pi"}\n')
    assert record.status is ArtifactStatus.ACTIVE
    assert record.validation['validation_strength'] == 'explicit_test_contract'
    assert await runtime.execute_tool(record.name, {}) == {'text': 'pi'}
