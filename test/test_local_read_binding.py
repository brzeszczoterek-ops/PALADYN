import asyncio
import json
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.autonomy import SemanticIntent, TaskContract
from v_core.memory.session import Session
from v_core.llm.llm import LLMResponse
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile


DEFINITIONS = [{"type": "function", "function": {
    "name": "read_file", "parameters": {"type": "object",
        "properties": {"path": {"type": "string"}}, "required": ["path"]},
}}]
CONTRACT = TaskContract(requires_file_read=True, required_tools=("read_file",))


@pytest.mark.parametrize("prompt", [
    "Użyj read_file: test/fixtures/local_tool_probe.txt. Podaj wynik.",
    "Use read_file to read test/fixtures/local_tool_probe.txt.",
    "Lee test/fixtures/local_tool_probe.txt con read_file.",
    "Use read_file to read README.md.",
    'Use read_file to read "docs/My Report.md".',
])
def test_read_path_binding_is_structural(prompt):
    expected = (
        "docs/My Report.md"
        if "My Report" in prompt
        else "README.md"
        if "README" in prompt
        else "test/fixtures/local_tool_probe.txt"
    )
    assert Agent._runtime_grounded_required_tool_request(
        prompt, CONTRACT, DEFINITIONS, []
    ) == ("read_file", {"path": expected})


@pytest.mark.parametrize("prompt", [
    'Use read_file. Say "testing now".',
    "Use read_file on a/one.txt or a/two.txt.",
    "Use read_file on ../secret.txt.",
    "Use read_file https://example.org/docs/file.txt.",
    "Use read_file on README.md or CHANGELOG.md.",
    'Use read_file on "docs/My Report.md" or README.md.',
    'Use read_file on "../secret.txt" or README.md.',
    "Use read_file HTTPS://example.org/docs/file.txt.",
])
def test_read_binding_does_not_guess(prompt):
    assert Agent._runtime_grounded_required_tool_request(prompt, CONTRACT, DEFINITIONS, []) is None


def test_failed_literal_read_is_not_automatically_replayed():
    prompt = "Use read_file on test/fixtures/local_tool_probe.txt."
    failures = [{"tool": "read_file", "arguments": {"path": "test/fixtures/local_tool_probe.txt"}}]
    assert Agent._runtime_grounded_required_tool_request(prompt, CONTRACT, DEFINITIONS, [], failures) is None


def test_literal_read_is_bound_when_contract_requires_capability_not_provider():
    contract = TaskContract(requires_file_read=True, requires_evidence_report=True)
    prompt = (
        "Analyze test/fixtures/code_analysis_probe.py without changing it. "
        "Do not use the network."
    )

    assert Agent._runtime_grounded_required_tool_request(
        prompt,
        contract,
        DEFINITIONS,
        [],
    ) == (
        "read_file",
        {"path": "test/fixtures/code_analysis_probe.py"},
    )


def test_incomplete_read_report_deduplicates_missing_requirement():
    answer = Agent._incomplete_task_answer(
        ["read_file", "read_file"],
        [{"tool": "read_file", "error": "missing"}],
    )
    assert "evidence: read_file." in answer
    assert "read_file, read_file" not in answer


def test_clarification_gate_requires_real_unresolved_argument():
    assert Agent._is_required_argument_clarification(
        "Use read_file on README.md or CHANGELOG.md.",
        "Which file: README.md or CHANGELOG.md?",
        CONTRACT,
        DEFINITIONS,
    )
    assert Agent._is_required_argument_clarification(
        "Use read_file on README.md or CHANGELOG.md.",
        "Pick one: README or CHANGELOG?",
        CONTRACT,
        DEFINITIONS,
    )
    assert Agent._is_required_argument_clarification(
        "Use read_file on README.md or CHANGELOG.md.",
        "Which file should I read?",
        CONTRACT,
        DEFINITIONS,
    )
    assert not Agent._is_required_argument_clarification(
        "Use read_file on README.md.",
        "Should I read README.md?",
        CONTRACT,
        DEFINITIONS,
    )
    assert not Agent._is_required_argument_clarification(
        "Use read_file on README.md or CHANGELOG.md.",
        "What should I do?",
        CONTRACT,
        DEFINITIONS,
    )


@pytest.mark.asyncio
async def test_missing_local_file_stops_with_recorded_failure(tmp_path):
    target = tmp_path / "missing.txt"
    calls = []

    class Tools:
        async def openai_tool_definitions(self):
            return DEFINITIONS

        async def call(self, tool, arguments):
            calls.append((tool, arguments))
            return target.read_text(encoding="utf-8")

    class Model:
        config = SimpleNamespace(context=8192)

        async def respond(self, **kwargs):
            raise AssertionError("A failed literal read must not generate a success report")

    class Memory:
        session = Session()

        async def process(self, *args, **kwargs):
            pass

    agent = object.__new__(Agent)
    agent.llm = Model()
    agent.tools = Tools()
    agent.memory = Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    answer = await agent._run_agent_loop(
        f"Use read_file to read {target}. Read-only. Report the contents."
    )
    await asyncio.gather(*agent._memory_tasks)
    assert len(calls) == 1
    assert "FileNotFoundError" in answer
    assert "Nothing is running in the background" in answer
    checkpoint = json.loads(next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    ).read_text(encoding="utf-8"))
    assert checkpoint["status"] != "completed"
    assert len(checkpoint["tool_calls"]) == 1
    assert checkpoint["tool_calls"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_ambiguous_local_read_accepts_grounded_clarification(tmp_path):
    class IntentRouter:
        async def classify(self, *args, **kwargs):
            return SemanticIntent(
                message_clear=True,
                action_requested=True,
                capabilities=("file_read",),
            )

    class Tools:
        async def openai_tool_definitions(self):
            return DEFINITIONS

        async def call(self, tool, arguments):
            raise AssertionError("ambiguous input must not execute a tool")

    class Model:
        config = SimpleNamespace(context=8192)

        async def respond(self, **kwargs):
            return LLMResponse(content="Which file should I read: README.md or CHANGELOG.md?")

    class Memory:
        session = Session()

        async def process(self, *args, **kwargs):
            pass

    agent = object.__new__(Agent)
    agent.intent_router = IntentRouter()
    agent.llm = Model()
    agent.tools = Tools()
    agent.memory = Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    answer = await agent._run_agent_loop(
        "Use read_file on README.md or CHANGELOG.md. Ask me which one before reading."
    )
    await asyncio.gather(*agent._memory_tasks)
    assert answer == "Which file should I read: README.md or CHANGELOG.md?"
    checkpoint = json.loads(next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    ).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "awaiting_owner"
    assert checkpoint["owner_checkpoint"]["missing"] == ["read_file"]
    assert checkpoint["owner_checkpoint"]["accepted_commands"] == [
        "reply with the missing information",
        "/stop",
    ]
