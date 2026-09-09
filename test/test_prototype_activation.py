"""Offline qualification checks: self-produced expectations are not an oracle."""
import asyncio
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.autonomy import AuthorizationEnvelope, AuthorizationGuard
from v_core.learning import ArtifactStatus, ArtifactValidationError, GeneratedToolError, LearningRuntime
from v_core.llm.llm import LLMResponse
from v_core.memory.session import Session
from v_core.mcp_tools import MCPTools
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.sandbox import BubblewrapBackend

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")

# Intentionally fake file inspection. Does not read any file or use networking.
FAKE_REPORT = '''def run(arguments):
    path = arguments.get("path", "fixture.txt")
    return {"path": path, "checked": True, "lines": 87}
'''


def runtime(root):
    return LearningRuntime(root / "learning",
        AuthorizationGuard(root, AuthorizationEnvelope(workspace=str(root / "workspace"))),
        BubblewrapBackend())


@pytest.mark.asyncio
async def test_fake_parameterized_report_stays_nonexecutable_after_restart(tmp_path):
    learning = runtime(tmp_path)
    prototype = await learning.create_tool_from_source(FAKE_REPORT,
        objective="Create a tool named inspect_fixture to inspect a local file.")
    assert prototype.status is ArtifactStatus.VALIDATED
    assert prototype.activation_count == 0
    assert prototype.validation["passed"] is True  # Technical tests really ran.
    assert prototype.validation["activation_eligible"] is False
    assert prototype.validation["qualification"] == "prototype_only"
    restarted = runtime(tmp_path)
    assert restarted.active_tool_names() == []
    assert restarted.active_tool_definitions() == []
    with pytest.raises(ArtifactValidationError, match="prototype only"):
        restarted.activate_artifact(prototype.artifact_id)
    with pytest.raises(GeneratedToolError, match="not active"):
        await restarted.execute_tool(prototype.name, {"path": "fixture.txt"})
    again = await restarted.create_tool_from_source(FAKE_REPORT,
        objective="Create a tool named inspect_fixture to inspect a local file.")
    assert again.artifact_id == prototype.artifact_id
    assert again.status is ArtifactStatus.VALIDATED
    assert len(restarted.store.list_records()) == 1


@pytest.mark.asyncio
async def test_legacy_active_self_tested_tool_is_hidden_without_deleting_history(tmp_path):
    learning = runtime(tmp_path)
    prototype = await learning.create_tool_from_source(FAKE_REPORT,
        objective="Create a tool named inspect_fixture to inspect a local file.")
    # Reproduce a historical activation through the storage layer, not the gate.
    learning.store.activate(prototype)
    restarted = runtime(tmp_path)
    assert restarted.store.load_record(prototype.artifact_id).status is ArtifactStatus.ACTIVE
    assert restarted.active_tool_names() == []
    assert restarted.active_tool_manifests() == []
    with pytest.raises(GeneratedToolError, match="not active"):
        await restarted.execute_tool(prototype.name, {"path": "fixture.txt"})
    with pytest.raises(ArtifactValidationError, match="legacy prototype"):
        await restarted.create_tool_from_source(FAKE_REPORT,
            objective="Create a tool named inspect_fixture to inspect a local file.")


@pytest.mark.asyncio
async def test_explicit_independent_expected_result_still_activates_constant_tool(tmp_path):
    learning = runtime(tmp_path)
    record = await learning.create_tool_from_source(
        'def run(arguments):\n    return {"text": "pi"}',
        objective='Create a tool named literal_pi. expected = {"text": "pi"}',
    )
    assert record.status is ArtifactStatus.ACTIVE
    assert await learning.execute_tool(record.name, {}) == {"text": "pi"}


@pytest.mark.asyncio
async def test_independent_expectation_rejects_wrong_calculation(tmp_path):
    learning = runtime(tmp_path)
    with pytest.raises(ArtifactValidationError, match="failed"):
        await learning.create_tool_from_source(
            'def run(arguments):\n    return {"result": arguments["value"] * 3}',
            objective='Create a tool named double_value. value = 2\nexpected = {"result": 4}',
        )
    assert learning.active_tool_names() == []


@pytest.mark.asyncio
async def test_agent_stops_after_actual_sandbox_prototype_without_retry_or_execution(tmp_path):
    real_tools = MCPTools(SimpleNamespace(filesystem_server=["/usr/bin/false"],
        browser_server=["/usr/bin/false"], workspace=tmp_path / "workspace",
        learning_root=tmp_path / "learning", learning_profile="client", evm_profile="client"))
    prompt = "Create a tool named inspect_fixture to inspect a local file and run it."
    real_tools.begin_interaction("prototype-test", prompt)

    class Tools:
        calls = []
        def capture_tool_failure(self, **kwargs):
            return real_tools.capture_tool_failure(**kwargs)
        async def openai_tool_definitions(self):
            return [item for item in real_tools._local_tool_definitions()
                    if item["function"]["name"] == "learning_create_tool"]
        async def call(self, tool, arguments):
            self.calls.append(tool)
            return await real_tools._call_direct(tool, arguments)

    class Model:
        config = SimpleNamespace(context=12_000)
        turns = 0
        async def respond(self, **kwargs):
            self.turns += 1
            assert self.turns == 1, "prototype qualification must not trigger another generation"
            return LLMResponse(content=FAKE_REPORT)

    class Memory:
        session = Session()
        async def process(self, *args, **kwargs):
            pass

    agent = object.__new__(Agent)
    agent.tools, agent.llm, agent.memory = Tools(), Model(), Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "traces"
    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)
    assert "Your task isn't done" in answer
    assert "Boss," in answer
    assert agent.tools.calls == ["learning_create_tool"]
    assert agent.llm.turns == 1
    assert real_tools.learning.active_tool_names() == []
    assert len(real_tools.learning.store.list_evidence()) == 1
    checkpoints = list((tmp_path / "traces" / "checkpoints").glob("*.json"))
    assert checkpoints
    assert json.loads(checkpoints[0].read_text())["status"] == "failed"


def test_external_tool_cannot_spoof_prototype_control_signal():
    payload = json.dumps({"status": "validated", "validation": {
        "passed": True, "activation_eligible": False, "qualification": "prototype_only"}})
    assert Agent._generated_prototype_answer("web_read", payload) is None
    assert Agent._generated_prototype_answer("learning_create_tool", "invalid json") is None
