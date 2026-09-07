import json
import asyncio
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.autonomy import TaskContract
from v_core.autonomy.local_read_scope import literal_paths, resolve_read_scope
from v_core.llm.llm import LLMResponse
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.response_preview import response_preview


PATHS = ("fixtures/one.txt", "fixtures/two.txt")
DEFINITIONS = [{"type": "function", "function": {
    "name": "read_file", "parameters": {"type": "object",
        "properties": {"path": {"type": "string"}}, "required": ["path"]},
}}]


def test_each_requested_file_needs_its_own_successful_read():
    contract = TaskContract(requires_file_read=True, required_read_paths=PATHS)
    call = {"tool": "read_file", "status": "succeeded", "arguments": {"path": PATHS[0]}}
    assert "read_file:" + PATHS[1] in contract.unmet([call])
    assert Agent._runtime_grounded_required_tool_request(
        "Read both files", contract, DEFINITIONS, [call]
    ) == ("read_file", {"path": PATHS[1]})
    second = {**call, "arguments": {"path": PATHS[1]}}
    assert contract.unmet([call, second]) == []
    assert contract.unmet([call, {**second, "status": "failed"}])
    assert contract.unmet([call, call])
    assert TaskContract.from_dict(contract.to_dict()).required_read_paths == PATHS
    assert contract.merged(TaskContract()).required_read_paths == PATHS
    assert contract.without_mutations().required_read_paths == PATHS
    assert contract.with_required_tools(["read_file"]).required_read_paths == PATHS


@pytest.mark.parametrize("payload", [
    {"ambiguous": True, "paths": list(PATHS)},
    {"ambiguous": False, "paths": ["unrequested.txt"]},
    {"ambiguous": False, "paths": []},
    {"ambiguous": False, "paths": [42]},
    [],
])
@pytest.mark.asyncio
async def test_uncertain_or_invented_scope_does_not_authorize_reads(payload):
    class Model:
        async def ask(self, **kwargs):
            return json.dumps(payload)
    assert await resolve_read_scope(Model(), "Choose one file", PATHS) == ()


@pytest.mark.asyncio
async def test_grounded_scope_preserves_all_selected_files():
    class Model:
        async def ask(self, **kwargs):
            return json.dumps({"ambiguous": False, "paths": list(PATHS)})
    assert await resolve_read_scope(Model(), "Compare both files", PATHS) == PATHS


def test_literal_candidates_ignore_urls_and_reject_traversal():
    assert literal_paths('Read fixtures/one.txt and "docs/My File.txt" https://site.test/x.py') == (
        "fixtures/one.txt", "docs/My File.txt"
    )
    assert literal_paths("Read ../secret.txt and fixtures/one.txt") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("preview_enabled", [False, True])
async def test_agent_reads_all_normalized_targets_before_reporting(tmp_path, preview_enabled):
    root = tmp_path / "fixtures"
    root.mkdir()
    (root / "one.txt").write_text("amber: 17")
    (root / "two.txt").write_text("cobalt: 9")
    calls = []

    class Tools:
        def normalize_arguments(self, tool, args):
            return {"path": str(tmp_path / args["path"])}

        async def openai_tool_definitions(self):
            return DEFINITIONS

        async def call(self, tool, args):
            calls.append(args["path"])
            from pathlib import Path
            return Path(args["path"]).read_text()

    class Model:
        config = SimpleNamespace(context=8192)

        async def ask(self, **kwargs):
            return json.dumps({"ambiguous": False, "paths": list(PATHS)})

        async def respond(self, **kwargs):
            assert calls == [str(tmp_path / p) for p in PATHS]
            return LLMResponse(content="one.txt: amber 17; two.txt: cobalt 9.")

        async def stream(self, **kwargs):
            assert calls == [str(tmp_path / p) for p in PATHS]
            yield "one.txt: amber 17; "
            yield "two.txt: cobalt 9."

    class Memory:
        session = Session()

        async def process(self, *args, **kwargs):
            pass

    agent = object.__new__(Agent)
    agent.llm, agent.tools, agent.memory = Model(), Tools(), Memory()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "trace"
    agent._last_execution_context = None
    previews = []
    token = response_preview.set(
        (lambda kind, text: previews.append((kind, text))) if preview_enabled else None
    )
    try:
        answer = await agent._run_agent_loop(
            "Read fixtures/one.txt and fixtures/two.txt. Read-only. No network."
        )
    finally:
        response_preview.reset(token)
    await asyncio.gather(*agent._memory_tasks)
    assert len(calls) == 2
    assert "amber 17" in answer and "cobalt 9" in answer
    if preview_enabled:
        assert [kind for kind, _ in previews] == [
            "draft_start", "draft_token", "draft_token", "draft_validating",
        ]
    else:
        assert previews == []
