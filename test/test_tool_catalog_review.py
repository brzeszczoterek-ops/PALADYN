import json
from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.autonomy import SemanticIntent
from v_core.memory.session import Session
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.tool_catalog_review import review_catalog


def definition(name, description="Read metadata", properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
    }}


def test_catalog_findings_are_grounded_in_schema_not_model_claims():
    text, evidence = review_catalog([
        definition("read_file", properties={"path": {"type": "string"}}, required=["path", "missing"]),
        definition("memory_recall", description=""),
    ], "Review tool metadata")
    assert "required lists undeclared fields: missing" in text
    assert "path: parameter meaning is undocumented" in text
    assert "description is missing" in text
    assert evidence["functional_tests"] == 0
    assert "not their implementation" in text


def test_catalog_explicit_subset_and_empty_catalog():
    text, evidence = review_catalog([definition("read_file"), definition("memory_recall")], "Describe read_file")
    assert evidence["count"] == 1
    assert "memory_recall" not in text
    assert review_catalog([], "review")[1]["status"] == "empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", [
    "Przejrzyj dostępne narzędzia i podaj propozycje usprawnień ich opisów.",
    "Review available tool descriptions and suggest improvements.",
    "Revisa las descripciones de las herramientas disponibles.",
])
async def test_metadata_review_reads_catalog_without_executing_tools(tmp_path, prompt):
    class Tools:
        async def openai_tool_definitions(self):
            return [definition("read_file")]

        async def call(self, *args, **kwargs):
            raise AssertionError("metadata review must not execute tools")

    class Router:
        async def classify(self, *args, **kwargs):
            return SemanticIntent(action_requested=True, capabilities=("tool_catalog",), requires_report=True)

    agent = object.__new__(Agent)
    agent.tools = Tools()
    agent.intent_router = Router()
    agent.memory = SimpleNamespace(session=Session())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    answer = await agent._run_agent_loop(prompt)
    assert "inspected 1 discovered tool definitions" in answer
    events = [json.loads(line) for line in next((tmp_path / "journal").glob("*.jsonl")).read_text().splitlines()]
    observed = [e for e in events if e["event"] == "tool_catalog_review"]
    assert observed[0]["data"]["functional_tests"] == 0


def test_catalog_capability_parses():
    intent = SemanticIntent.parse(json.dumps({"action_requested": True, "capabilities": ["tool_catalog"], "requires_report": True}))
    assert intent and intent.capabilities == ("tool_catalog",)
