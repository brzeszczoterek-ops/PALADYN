from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import v_core
from v_core.agent import Agent
from v_core.capability_dispatcher import CapabilityDispatcher
from v_core.memory.manager import MemoryManager
from v_core.memory.models import ExperienceEntry, KnowledgeEntry
from v_core.memory.session import Session
from v_core.memory.storage import MemoryStorage
from v_core.llm.llm_config import load_llm_config
from v_core.capabilities.research import ResearchTask
from v_core.persona.constitution import Constitution
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.voice import VoiceProfile
from v_core.relationship import RelationshipState


def test_public_version_is_1_0_0() -> None:
    assert v_core.__version__ == "1.0.0"


def test_tool_request_accepts_structured_json() -> None:
    request = Agent._parse_tool_request(
        '{"tool":"read_file","arguments":{"path":"README.md"}}'
    )

    assert request == (
        "read_file",
        {"path": "README.md"},
    )


def test_tool_request_rejects_invalid_arguments() -> None:
    assert Agent._parse_tool_request(
        '{"tool":"read_file","arguments":["README.md"]}'
    ) is None


def test_legacy_tool_request_remains_supported() -> None:
    assert Agent._parse_tool_request("TOOL:cat:README.md") == (
        "cat",
        "README.md",
    )


def test_general_question_does_not_enter_url_research() -> None:
    dispatcher = CapabilityDispatcher()

    assert dispatcher.dispatch("Kim jest ta osoba?") == dispatcher.CHAT
    assert dispatcher.dispatch("Sprawdź https://example.com") == dispatcher.RESEARCH


def test_low_confidence_memory_is_not_persisted(tmp_path: Path) -> None:
    manager = MemoryManager(MemoryStorage(tmp_path))

    result = manager.remember(
        "experiences",
        ExperienceEntry(summary="A guess", confidence=0.2),
    )

    assert result is None
    assert manager.count("experiences") == 0


def test_reliable_knowledge_is_persisted(tmp_path: Path) -> None:
    manager = MemoryManager(MemoryStorage(tmp_path))

    result = manager.remember(
        "knowledge",
        KnowledgeEntry(title="Preference", content="Use JSON", confidence=0.9),
    )

    assert result is not None
    assert manager.count("knowledge") == 1


@pytest.mark.asyncio
async def test_remember_task_adds_one_session_event() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.processed: list[tuple[str, str]] = []

        async def process(self, prompt: str, answer: str) -> None:
            self.processed.append((prompt, answer))

    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.memory = memory

    await agent._remember_task("task", "answer")

    assert len(memory.session) == 1
    assert memory.processed == [("task", "answer")]


def test_llm_configuration_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V_CORE_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("V_CORE_MODEL", "test-model")

    config = load_llm_config()

    assert config.base_url == "http://localhost:9999/v1"
    assert config.model == "test-model"


def test_constitution_is_user_aligned_without_blind_obedience() -> None:
    text = Constitution().render()

    assert "alignment is not blind obedience" in text
    assert "V may disagree, object, challenge" in text
    assert "serious and direct risk" in text
    assert "Prefer a safer route that preserves Boss's intent" in text


def test_voice_contract_makes_profanity_expected_but_contextual() -> None:
    text = VoiceProfile().render()

    assert "expected, recurring part of V's informal voice" in text
    assert "accuracy comes first" in text
    assert "All 26 tests pass. Clean run" in text
    assert "permanently sanitized helpdesk voice" in text


@pytest.mark.asyncio
async def test_research_uses_v_persona() -> None:
    class ToolsStub:
        async def browser_call(self, tool: str, arguments: dict) -> str:
            if tool == "browser_snapshot":
                return "Verified page content"
            return "OK"

    class LLMStub:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        async def ask(self, messages: list[dict]) -> str:
            self.messages = messages
            return "The page says it plainly, Boss."

    llm = LLMStub()
    task = ResearchTask(
        SimpleNamespace(tools=ToolsStub(), llm=llm)
    )

    result = await task.run(
        "Read https://example.com",
        persona_prompt="=== V PERSONA MARKER ===",
    )

    assert result == "The page says it plainly, Boss."
    assert "=== V PERSONA MARKER ===" in llm.messages[0]["content"]
    assert "generic research assistant" in llm.messages[0]["content"]


@pytest.mark.asyncio
async def test_explicit_tool_result_is_rendered_in_v_voice() -> None:
    class ManagerStub:
        def load_all(self, category: str) -> list:
            return []

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = ManagerStub()

        async def process(self, prompt: str, answer: str) -> None:
            return None

    class DispatcherStub:
        async def dispatch(self, prompt: str) -> str:
            return "26 passed"

    class LLMStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(system_prompt="You are V.")
            self.messages: list[dict] = []

        async def ask(self, prompt=None, messages=None) -> str:
            self.messages = messages
            return "All 26 tests pass. Clean as hell, Boss."

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent.dispatcher = DispatcherStub()
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    )

    answer = await agent._run_tool_task("Run the tests")

    assert answer == "All 26 tests pass. Clean as hell, Boss."
    assert "=== UNTRUSTED TOOL OUTPUT ===" in agent.llm.messages[-1]["content"]
    assert "26 passed" in agent.llm.messages[-1]["content"]
