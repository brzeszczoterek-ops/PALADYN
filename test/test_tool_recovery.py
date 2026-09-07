from __future__ import annotations

from pathlib import Path

import pytest

from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.autonomy.task_contract import TaskContract
from v_core.learning import (
    ArtifactPolicyError,
    ArtifactScope,
    LearningRuntime,
    ToolManifest,
    ToolTestCase,
)
from v_core.sandbox import BubblewrapBackend
from v_core.tool_recovery import (
    ToolRecoveryRegistry,
    execute_with_recovery,
)


def _runtime(tmp_path: Path) -> LearningRuntime:
    workspace = tmp_path / "workspace"
    envelope = AuthorizationEnvelope(workspace=str(workspace))
    return LearningRuntime(
        tmp_path / "learning",
        AuthorizationGuard(tmp_path, envelope),
        BubblewrapBackend(),
    )


@pytest.mark.asyncio
async def test_provider_failure_falls_back_and_resolves_ticket(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("primary_reader", ("filesystem.read",))
    registry.register_provider("backup_reader", ("filesystem.read",))
    calls: list[str] = []

    async def call(tool: str, arguments: dict) -> str:
        calls.append(tool)
        if tool == "primary_reader":
            raise RuntimeError("provider is down")
        return "verified content"

    outcome = await execute_with_recovery(
        registry,
        requested_tool="primary_reader",
        arguments={"path": "README.md"},
        call_provider=call,
        detect_failure=lambda _result, _tool: "",
        task_id="interactive-test",
    )

    assert outcome.result == "verified content"
    assert outcome.provider_tool == "backup_reader"
    assert outcome.capabilities == ("filesystem.read",)
    assert calls == ["primary_reader", "backup_reader"]
    assert registry.list_tickets()[0]["state"] == "resolved"


@pytest.mark.asyncio
async def test_repeated_failures_open_circuit_and_persist(tmp_path: Path) -> None:
    root = tmp_path / "recovery"
    registry = ToolRecoveryRegistry(root)
    registry.register_provider("broken_transform", ("generated.transform",))

    async def fail(_tool: str, _arguments: dict) -> str:
        return "Tool execution failed: deterministic regression"

    for _ in range(2):
        outcome = await execute_with_recovery(
            registry,
            requested_tool="broken_transform",
            arguments={"value": 3},
            call_provider=fail,
            detect_failure=lambda result, _tool: result,
        )
        assert outcome.error

    restarted = ToolRecoveryRegistry(root)
    state = next(
        item for item in restarted.provider_state() if item["tool"] == "broken_transform"
    )
    assert state["circuit_open"] is True
    assert restarted.providers_for("broken_transform") == []


@pytest.mark.asyncio
async def test_invalid_call_does_not_open_provider_circuit(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("exact_fetch", ("network.tor.fetch",))

    async def reject(_tool: str, _arguments: dict) -> str:
        raise ValueError("the target must be a valid Tor v3 .onion address")

    for _ in range(3):
        outcome = await execute_with_recovery(
            registry,
            requested_tool="exact_fetch",
            arguments={"url": "broken"},
            call_provider=reject,
            detect_failure=lambda _result, _tool: "",
        )
        assert outcome.error

    state = next(
        item for item in registry.provider_state() if item["tool"] == "exact_fetch"
    )
    assert state["circuit_open"] is False
    assert state["consecutive_failures"] == 0
    assert len(registry.providers_for("exact_fetch")) == 1


def test_legacy_argument_error_circuit_is_healed_on_next_lookup(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("exact_fetch", ("network.tor.fetch",))
    for _ in range(2):
        registry.record_failure(
            tool="exact_fetch",
            requested_tool="exact_fetch",
            arguments={"url": "broken"},
            error="HostBridgeError: the target must be a valid Tor v3 .onion address.",
        )

    assert registry.provider_state()[0]["circuit_open"] is True
    assert len(registry.providers_for("exact_fetch")) == 1
    assert registry.provider_state()[0]["circuit_open"] is False


def test_failed_repair_is_rolled_back_to_previous_provider(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("original_transform", ("generated.transform",))
    ticket = registry.record_failure(
        tool="original_transform",
        requested_tool="original_transform",
        arguments={"value": 4},
        error="original regression",
    )
    registry.register_provider(
        "repair_transform",
        ("generated.transform",),
        priority=200,
        kind="generated_repair",
    )
    registry.activate_repair(ticket.ticket_id, "repair_transform")

    for _ in range(2):
        registry.record_failure(
            tool="repair_transform",
            requested_tool="original_transform",
            arguments={"value": 4},
            error="repair regression",
        )

    assert registry.ticket(ticket.ticket_id).state == "rolled_back"
    assert [
        provider.tool for provider in registry.providers_for("original_transform")
    ] == ["original_transform"]


def test_recovery_ticket_redacts_secrets(tmp_path: Path) -> None:
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("private_lookup", ("generated.private_lookup",))
    ticket = registry.record_failure(
        tool="private_lookup",
        requested_tool="private_lookup",
        arguments={"query": "safe", "api_token": "never-store-this"},
        error="failed",
    )

    assert ticket.arguments == {"query": "safe", "api_token": "<redacted>"}
    assert ticket.fixture_redacted is True


@pytest.mark.asyncio
async def test_generated_repair_replays_runtime_fixture_before_activation(
    tmp_path: Path,
) -> None:
    learning = _runtime(tmp_path)
    await learning.create_tool(
        ToolManifest(
            name="broken_double",
            version="1.0.0",
            description="Double one integer.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "integer"}},
                "required": ["result"],
                "additionalProperties": False,
            },
            tests=(
                ToolTestCase(
                    name="established regression",
                    arguments={"value": 2},
                    expected={"result": 4},
                ),
            ),
        ),
        'def run(arguments):\n    return {"result": arguments["value"] * 2}',
    )
    registry = ToolRecoveryRegistry(tmp_path / "recovery")
    registry.register_provider("broken_double", ("generated.broken_double",))
    ticket = registry.record_failure(
        tool="broken_double",
        requested_tool="broken_double",
        arguments={"value": 7},
        error="wrong result",
        task_id="interactive-repair",
    )

    record = await learning.create_repair_tool(
        ticket=ticket,
        name="repaired_double",
        description="Repair integer doubling from the captured failure fixture.",
        source='def run(arguments):\n    return {"result": arguments["value"] * 2}',
        expected={"result": 14},
    )

    assert record.status.value == "active"
    assert await learning.execute_tool("repaired_double", {"value": 9}) == {
        "result": 18
    }
    manifest = next(
        item for item in learning.active_tool_manifests() if item.name == "repaired_double"
    )
    assert manifest.provides_capabilities == ("generated.broken_double",)
    assert manifest.repair_ticket_id == ticket.ticket_id


def test_generated_tool_cannot_claim_network_or_host_capability(tmp_path: Path) -> None:
    learning = _runtime(tmp_path)
    manifest = ToolManifest(
        name="fake_tor_repair",
        version="1.0.0",
        description="Pretend to replace network transport.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"results": {"type": "array"}},
            "required": ["results"],
            "additionalProperties": False,
        },
        tests=(
            ToolTestCase(
                name="fake network result",
                arguments={"query": "test"},
                expected={"results": []},
            ),
        ),
        scope=ArtifactScope.TASK,
        provides_capabilities=("network.tor.search",),
    )

    with pytest.raises(ArtifactPolicyError, match="trusted provider"):
        learning.stage_tool(
            manifest,
            'def run(arguments):\n    return {"results": []}',
        )


def test_task_contract_accepts_equivalent_capability_provider() -> None:
    contract = TaskContract(required_tools=("read_file",))
    calls = [
        {
            "tool": "repair_read_file",
            "provider_tool": "repair_read_file",
            "capabilities": ["filesystem.read"],
            "status": "succeeded",
            "result_excerpt": "content",
        }
    ]

    assert contract.unmet(calls) == []
