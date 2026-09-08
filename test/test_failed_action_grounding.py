from types import SimpleNamespace

import pytest

from v_core.agent import Agent
from v_core.autonomy import TaskContract


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt,force_empty_action", [
    ("Świetnie, posłuchaj V, to w takim razie teraz weź i każdy stór, który jest w tej chwili wybrakowany, albo w którym czegoś nie ma, wyszukaj to czego mu brakuje i dołącz do tego, napraw je.", False),
    ("Fix the things we discussed and tell me when they work.", True),
    ("Repara lo que falta y comprueba que funciona.", True),
])
async def test_rejected_action_with_empty_contract_never_generates_success(tmp_path, monkeypatch, prompt, force_empty_action):
    class Router:
        last_failure_reason = "current_message_grounding"
        last_response = ""

        async def classify(self, *args, **kwargs):
            return None

    class NoGeneration:
        config = SimpleNamespace(context=12000)

        async def respond(self, **kwargs):
            raise AssertionError("Rejected empty action must not generate a success report")

    class NoTools:
        def begin_interaction(self, *args, **kwargs):
            pass

        async def openai_tool_definitions(self):
            raise AssertionError("Rejected empty action must not invoke tools")

    agent = object.__new__(Agent)
    agent.llm = NoGeneration()
    agent.tools = NoTools()
    agent.intent_router = Router()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    # Reproduce the structural failure: lexical action survives, but semantic
    # grounding and the resulting execution contract do not. Language is not
    # part of the safety decision.
    if force_empty_action:
        monkeypatch.setattr(TaskContract, "from_prompt", classmethod(lambda cls, _: cls()))
        monkeypatch.setattr(agent, "_requests_runtime_action", lambda *args: True)
    answer = await agent._run_agent_loop(prompt)
    assert "No tools ran" in answer
    assert "no repairs or tests were performed" in answer
