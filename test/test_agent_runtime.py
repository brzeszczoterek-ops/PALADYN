from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError

import v_core
from v_core.agent import Agent
from v_core.autonomy import ContextWindowManager, SemanticIntent, TaskContract
from v_core.autonomy.agent_trace import AgentTaskTrace
from v_core.capability_dispatcher import CapabilityDispatcher
from v_core.memory.manager import MemoryManager
from v_core.memory.models import ExperienceEntry, KnowledgeEntry, MemorySource
from v_core.memory.reflection import Reflection
from v_core.memory.session import Session
from v_core.memory.storage import MemoryStorage
from v_core.mcp_tools import MCPToolExecutionError, MCPTools
from v_core.tools.filesystem import Filesystem, FilesystemToolError
from v_core.llm import LLM
from v_core.llm.llm import LLMResponse, LLMToolCall, repetition_start, trim_repetition
from v_core.llm.llm_config import load_llm_config
from v_core.capabilities.research import ResearchTask
from v_core.persona.constitution import Constitution
from v_core.persona.kernel import IdentityKernel
from v_core.persona.runtime import PersonaRuntime
from v_core.persona.language import (
    asks_user_to_use_english,
    explicitly_requests_non_english,
    looks_non_english,
)
from v_core.persona.voice import (
    VoiceProfile,
    looks_generic_assistant_voice,
    looks_sanitized_contempt,
)
from v_core.relationship import RelationshipState
import v_core.main as main_module


def test_public_version_is_1_5_0() -> None:
    assert v_core.__version__ == "1.5.0"


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


def test_tool_request_accepts_model_prose_before_trailing_json() -> None:
    request = Agent._parse_tool_request(
        "You're right. I'm doing it now.\n"
        '{"tool":"browser_navigate","arguments":'
        '{"url":"https://onehack.st"}}'
    )

    assert request == (
        "browser_navigate",
        {"url": "https://onehack.st"},
    )


def test_tool_request_accepts_trailing_fenced_json() -> None:
    request = Agent._parse_tool_request(
        "Calling the tool.\n```json\n"
        '{"tool":"read_file","arguments":{"path":"README.md"}}\n```'
    )

    assert request == ("read_file", {"path": "README.md"})


def test_tool_request_rejects_multiple_tool_objects_as_ambiguous() -> None:
    assert Agent._parse_tool_request(
        '{"tool":"read_file","arguments":{"path":"one"}}\n'
        '{"tool":"read_file","arguments":{"path":"two"}}'
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
    assert dispatcher.dispatch(
        "Wejdź na onehack.st i przejrzyj tę stronę"
    ) == dispatcher.RESEARCH


def test_capability_questions_and_tool_actions_use_multi_step_agent_loop() -> None:
    dispatcher = CapabilityDispatcher()

    assert dispatcher.dispatch(
        "V, czy potrafisz stworzyć własne narzędzia?"
    ) == dispatcher.CHAT
    assert dispatcher.dispatch("Przeczytaj plik README.md") == dispatcher.CHAT
    assert dispatcher.dispatch("Create a generated skill for CSV files") == (
        dispatcher.CHAT
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "Cześć V, jak tam się dzisiaj czujesz?",
        "Jak się dzisiaj czujesz?",
        "Co u ciebie?",
        "How are you feeling today?",
        "How's your day?",
    ],
)
def test_short_questions_about_v_state_are_light_conversation(prompt: str) -> None:
    assert Agent._is_light_conversation(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "Cześć V, sprawdź README.md",
        "Jak sprawdzić stronę example.com?",
        "How do you read the file README.md?",
        "Cześć, uruchom testy.",
    ],
)
def test_short_action_requests_do_not_bypass_agent_mode(prompt: str) -> None:
    assert not Agent._is_light_conversation(prompt)


@pytest.mark.asyncio
async def test_feeling_question_never_discovers_or_calls_tools() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.interactions = 0

        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            self.interactions += 1

        async def openai_tool_definitions(self):
            raise AssertionError("Light conversation must not discover MCP tools")

        async def call(self, *args, **kwargs):
            raise AssertionError("Light conversation must not call a tool")

    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "A little electrically restless, but good, Boss."

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()

        async def process(self, *args, **kwargs) -> None:
            raise AssertionError("Light conversation must not enter durable memory")

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.capabilities = CapabilityDispatcher()
    agent._memory_tasks = set()

    answer = await agent.run("Cześć V, jak tam się dzisiaj czujesz?")

    assert answer == "A little electrically restless, but good, Boss."
    assert agent.tools.interactions == 1
    assert len(agent.memory.session) == 1


def test_research_normalizes_bare_domain_to_https() -> None:
    research = ResearchTask(SimpleNamespace())

    assert research._extract_url("Przejrzyj onehack.st, proszę.") == (
        "https://onehack.st"
    )
    assert research._extract_url("Read https://example.com/docs.") == (
        "https://example.com/docs"
    )


def test_research_snapshot_is_bounded_for_small_model_context() -> None:
    snapshot = "A" * 100_000

    fitted = ResearchTask._fit_snapshot(
        snapshot,
        prompt="Inspect example.com",
        persona_prompt="V" * 8_000,
        persona_examples=[{"role": "assistant", "content": "example" * 100}],
        context_tokens=8_000,
    )

    assert len(fitted) < 12_000
    assert fitted.startswith("A" * 100)
    assert fitted.endswith("A" * 100)
    assert "PALADYN omitted" in fitted


def test_multi_page_research_keeps_evidence_from_every_visited_page() -> None:
    pages = [
        ("https://skills.example/", "ENTRY-" + "A" * 20_000),
        ("https://skills.example/skill/one", "DETAIL-ONE-" + "B" * 20_000),
        ("https://skills.example/skill/two", "DETAIL-TWO-" + "C" * 20_000),
    ]

    fitted = ResearchTask._fit_page_snapshots(
        pages,
        prompt="Extract skills from skills.example",
        persona_prompt="V" * 8_000,
        persona_examples=[],
        context_tokens=8_000,
    )

    assert "ENTRY-" in fitted
    assert "DETAIL-ONE-" in fitted
    assert "DETAIL-TWO-" in fitted
    assert fitted.count("PALADYN omitted") == 3


def test_research_selects_bounded_internal_skill_pages() -> None:
    snapshot = """
- link "Blog" [ref=e1]:
  - /url: /blog
- link "ECC" [ref=e2]:
  - /url: /skill/ecc
  - text: security research agent harness
- link "View details" [ref=e3]:
  - /url: /skill/ecc
- link "Web Recon" [ref=e4]:
  - /url: /skill/web-recon
  - text: web security and OSINT
- link "External" [ref=e5]:
  - /url: https://outside.example/skill/external
""".strip()

    candidates = ResearchTask._detail_page_candidates(
        snapshot,
        "https://skills.example/",
        "Extract web security skills from skills.example",
        limit=3,
    )

    assert candidates == [
        "https://skills.example/skill/web-recon",
        "https://skills.example/skill/ecc",
    ]


def test_research_replaces_hallucinated_metrics_with_verified_records() -> None:
    pages = [
        ("https://skillsllm.com/", "Marketplace entry page"),
        (
            "https://skillsllm.com/skill/ecc",
            """
- heading "ECC" [level=1] [ref=e1]
- paragraph [ref=e2]: The agent harness with skills, memory, security, and research-first development.
- generic [ref=e3]:
  - generic [ref=e4]: 242,219
  - generic [ref=e5]: stars
  - generic [ref=e6]: 36,702
  - generic [ref=e7]: forks
""".strip(),
        ),
        (
            "https://skillsllm.com/skill/hermes-agent",
            """
- heading "hermes-agent" [level=1] [ref=e8]
- paragraph [ref=e9]: The agent that grows with you
- generic [ref=e10]:
  - generic [ref=e11]: 234,437
  - generic [ref=e12]: stars
  - generic [ref=e13]: 47,175
  - generic [ref=e14]: forks
""".strip(),
        ),
    ]
    evidence = ResearchTask._format_page_snapshots(pages)
    hallucinated = (
        "**1. ECC (240k★)** Built for scale with clean documentation.\n"
        "**2. hermes-agent (500k★)** Performance under load is unmatched."
    )

    assert ResearchTask._contains_unsupported_facts(hallucinated, evidence)
    report = ResearchTask._verified_extraction_report(pages)

    assert report is not None
    assert "242,219 stars" in report
    assert "234,437 stars" in report
    assert "500k" not in report
    assert "performance under load" not in report.casefold()
    assert "memory, research, security, skill" in report


def test_research_replaces_whole_site_overclaim_with_verified_links() -> None:
    snapshot = """
- link "Mini Tools | Free AI Tools" [ref=e1] [cursor=pointer]:
  - /url: /mini-tools
- link "Random discussion" [ref=e2] [cursor=pointer]:
  - /url: /random
""".strip()

    answer = ResearchTask._enforce_single_page_scope(
        "I scraped the whole site and pulled out everything. It has code generators.",
        snapshot=snapshot,
        page_url="https://onehack.st/",
        prompt="Extract useful tools from onehack.st",
    )

    assert "inspected one snapshot" in answer
    assert "not the whole website" in answer
    assert "Mini Tools | Free AI Tools" in answer
    assert "https://onehack.st/mini-tools" in answer
    assert "code generators" not in answer
    assert "Random discussion" not in answer


def test_research_keeps_answer_that_respects_single_page_scope() -> None:
    answer = "The homepage snapshot visibly includes a link named Mini Tools."

    assert ResearchTask._enforce_single_page_scope(
        answer,
        snapshot="",
        page_url="https://example.com",
        prompt="Inspect example.com",
    ) == answer


def test_research_rejects_website_controls_misclassified_as_tools() -> None:
    snapshot = """
- link "Mini Tools | Free AI Tools" [ref=e1] [cursor=pointer]:
  - /url: /mini-tools
- link "SEO Shope" [ref=e2] [cursor=pointer]:
  - /url: https://seo.example/tools
""".strip()
    draft = """
### Tools
- Skip navigation links
- Search button
- Navigation menu toggle
- User card dialog
""".strip()

    answer = ResearchTask._enforce_single_page_scope(
        draft,
        snapshot=snapshot,
        page_url="https://onehack.st/",
        prompt="Extract useful tools and skills from onehack.st",
    )

    assert "ordinary website controls" in answer
    assert "Mini Tools | Free AI Tools" in answer
    assert "SEO Shope" in answer
    assert "Skip navigation" not in answer


def test_generation_loop_is_cut_at_second_long_repetition() -> None:
    phrase = "V keeps her identity and answers Boss with a direct point of view."
    text = f"Start here. {phrase} {phrase} {phrase}"

    trimmed = trim_repetition(text)

    assert trimmed == f"Start here. {phrase}"
    assert repetition_start("This answer contains no repeated generation loop.") is None


def test_short_phrase_requires_three_repetitions_before_cutting() -> None:
    assert trim_repetition("No hidden bullshit. No hidden bullshit.") == (
        "No hidden bullshit. No hidden bullshit."
    )
    assert trim_repetition(
        "No hidden bullshit. No hidden bullshit. No hidden bullshit."
    ) == "No hidden bullshit."


@pytest.mark.asyncio
async def test_llm_stream_stops_before_emitting_repeated_block() -> None:
    phrase = "V keeps her identity and answers Boss with a direct point of view."

    class Chunks:
        def __init__(self) -> None:
            self._chunks = iter((phrase, " " + phrase, " " + phrase))

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                content = next(self._chunks)
            except StopIteration as error:
                raise StopAsyncIteration from error
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
            )

    class Completions:
        async def create(self, **_kwargs):
            return Chunks()

    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(
        model="test",
        temperature=0.7,
        top_p=0.95,
    )
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )

    chunks = [
        chunk
        async for chunk in llm.stream(
            messages=[{"role": "user", "content": "Talk to me"}],
            max_tokens=128,
        )
    ]

    assert chunks == [phrase]


def test_session_history_keeps_newest_events_within_character_budget() -> None:
    session = Session()
    session.add("task", {"task": "old", "result": "x" * 20})
    session.add("task", {"task": "new", "result": "y" * 20})

    messages = session.messages(limit=10, max_characters=25)

    assert messages == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "y" * 20},
    ]


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
    await asyncio.gather(*agent._memory_tasks)

    assert len(memory.session) == 1
    assert memory.processed == [("task", "answer")]


@pytest.mark.asyncio
async def test_light_conversation_streams_and_skips_expensive_memory() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.processed = False

        async def process(self, prompt: str, answer: str) -> None:
            self.processed = True

    class LLMStub:
        async def stream(self, **kwargs):
            assert kwargs["max_tokens"] == 96
            messages = kwargs["messages"]
            assert "never insert a swear" in (
                messages[0]["content"]
            )
            assert "dangerous to badly designed systems" in messages[-2]["content"]
            for chunk in (
                "Hey, Boss. I'm doing pretty damn well — ",
                "awake, sharp, and ready. How are you?",
            ):
                yield chunk

    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.memory = memory
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    )
    emitted: list[str] = []

    answer = await agent._run_light_chat("Cześć, jak tam?", emitted.append)

    assert "".join(emitted) == answer
    assert answer.startswith("Hey, Boss")
    assert len(memory.session) == 1
    assert memory.processed is False


@pytest.mark.asyncio
async def test_streaming_guard_does_not_emit_polish_before_rewrite() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()

        async def process(self, prompt: str, answer: str) -> None:
            raise AssertionError("Greeting must not enter persistent memory")

    class LLMStub:
        async def stream(self, **kwargs):
            yield "Cześć, Boss. U mnie wszystko dobrze i jestem gotowa do pracy."

        async def ask(self, **kwargs) -> str:
            assert kwargs["max_tokens"] == 256
            return "Hey, Boss. I'm good and ready to work."

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    )
    emitted: list[str] = []

    answer = await agent._run_light_chat("Cześć, jak tam?", emitted.append)

    assert emitted == [answer]
    assert answer == "Hey, Boss. I'm good and ready to work."


@pytest.mark.asyncio
async def test_streaming_guard_never_exposes_internal_tool_json() -> None:
    class LLMStub:
        async def stream(self, **kwargs):
            yield '{"tool":"read_file",'
            yield '"arguments":{"path":"README.md"}}'

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    emitted: list[str] = []

    answer, did_emit = await agent._stream_guarded_english(
        [{"role": "user", "content": "Read README.md"}],
        emitted.append,
        max_tokens=128,
    )

    assert Agent._parse_tool_request(answer) == (
        "read_file",
        {"path": "README.md"},
    )
    assert did_emit is False
    assert emitted == []


@pytest.mark.asyncio
async def test_agent_executes_tool_after_prose_json_without_leaking_protocol(
    tmp_path: Path,
) -> None:
    class ManagerStub:
        def load_all(self, category: str) -> list:
            return []

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = ManagerStub()
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | str]] = []

        def local_tool_names(self) -> list[str]:
            return ["browser_navigate"]

        def render_matching_skills(self, prompt: str) -> str:
            return ""

        async def call(self, tool: str, arguments: dict | str) -> str:
            self.calls.append((tool, arguments))
            return "Page title: OneHack. URL: https://onehack.st"

    class LLMStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                system_prompt="You are V.",
                context=8_192,
            )
            self.turn = 0

        async def stream(self, **kwargs):
            self.turn += 1
            if self.turn == 1:
                yield "I'm initiating the scan now.\n"
                yield (
                    '{"tool":"browser_navigate","arguments":'
                    '{"url":"https://onehack.st"}}'
                )
            else:
                yield "I opened OneHack and verified the returned page title, Boss."

        async def ask(self, **kwargs) -> str:
            raise AssertionError("No rewrite should be necessary")

    memory = MemoryStub()
    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.memory = memory
    agent.tools = tools
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    emitted: list[str] = []

    answer = await agent._run_agent_loop("Inspect OneHack", emitted.append)
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == [
        ("browser_navigate", {"url": "https://onehack.st"})
    ]
    assert emitted == [answer]
    assert "{\"tool\"" not in "".join(emitted)
    assert memory.execution is not None
    assert memory.execution["successful_tool_count"] == 1
    checkpoints = list(
        (tmp_path / "autonomy" / "interactive" / "checkpoints").glob("*.json")
    )
    assert len(checkpoints) == 1


@pytest.mark.asyncio
async def test_agent_executes_tool_action_recovered_by_language_repair() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return "# PALADYN / V-Core"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    "Najpierw przeczytam wskazany plik.",
                    '{"tool":"read_file","arguments":{"path":"README.md"}}',
                    "The first heading is `# PALADYN / V-Core`, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Przeczytaj plik README.md")
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == [
        ("read_file", {"path": "README.md"}),
    ]
    assert answer == "The first heading is `# PALADYN / V-Core`, Boss."
    assert "{\"tool\"" not in answer


@pytest.mark.asyncio
async def test_agent_rejects_fake_background_work_and_requests_real_tool() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def call(self, tool: str, arguments: dict | str) -> str:
            self.calls += 1
            return "three records extracted"

    class LLMStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(context=8_192)
            self.responses = iter(
                (
                    "I'm initiating the systematic extraction now and will report back.",
                    '{"tool":"extract_records","arguments":{"url":"https://example.com"}}',
                    "I extracted three records. That's the completed result, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, prompt: str, answer: str) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.MAX_AGENT_STEPS = 4
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Extract the records")

    assert agent.tools.calls == 1
    assert answer.startswith("I extracted three records")
    conversation = agent.memory.session.messages()
    assert all("initiating the systematic" not in item["content"] for item in conversation)


@pytest.mark.asyncio
async def test_agent_blocks_fabricated_phone_call_and_remote_exploit(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        async def call(self, tool: str, arguments: dict | str) -> str:
            raise AssertionError("No real tool was requested")

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0
            self.responses = iter(
                (
                    (
                        "I'm going to call his number from a clean browser. If he "
                        "picks up, I'm using a remote desktop exploit."
                    ),
                    (
                        "He picked up. I used the remote desktop exploit to connect. "
                        "Now I'm on the line with him and I told him to call Boss."
                    ),
                    "I did not execute that action.",
                )
            )

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    async def passthrough(messages, answer: str) -> str:
        return answer

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.MAX_AGENT_STEPS = 4
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._enforce_english = passthrough

    answer = await agent._run_agent_loop("No to zadzwoń do niego")
    await asyncio.gather(*agent._memory_tasks)

    assert answer.startswith("I did not execute")
    assert "remote-system access" in answer
    assert "on the line" not in answer
    assert agent.memory.execution is not None
    assert agent.memory.execution["status"] == "blocked"
    assert agent.memory.execution["successful_tool_count"] == 0
    assert agent.llm.calls == 2

    checkpoint = next(
        (tmp_path / "autonomy" / "interactive" / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["tool_calls"] == []

    journal = checkpoint.parent.parent / "journal" / f"{checkpoint.stem}.jsonl"
    journal_text = journal.read_text(encoding="utf-8")
    assert "unsupported_execution_claim" in journal_text
    assert "task_blocked" in journal_text


@pytest.mark.asyncio
async def test_agent_can_recover_from_false_file_claim_with_real_tool(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | str]] = []

        async def call(self, tool: str, arguments: dict | str) -> str:
            self.calls.append((tool, arguments))
            return "Successfully wrote notes.txt"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    "I wrote the file, Boss.",
                    '{"tool":"write_file","arguments":{"path":"notes.txt","content":"ok"}}',
                    "I wrote the file, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    async def passthrough(messages, answer: str) -> str:
        return answer

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.MAX_AGENT_STEPS = 4
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._enforce_english = passthrough

    answer = await agent._run_agent_loop("Write notes.txt")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "I wrote the file, Boss."
    assert tools.calls == [
        ("write_file", {"path": "notes.txt", "content": "ok"})
    ]
    assert agent.memory.execution is not None
    assert agent.memory.execution["status"] == "completed"
    assert agent.memory.execution["successful_tool_count"] == 1


@pytest.mark.asyncio
async def test_website_claim_requires_navigation_and_snapshot_evidence() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def call(self, tool: str, arguments: dict | str) -> str:
            self.calls.append(tool)
            if tool == "browser_navigate":
                return "Navigated to https://onehack.st"
            if tool == "browser_snapshot":
                return "Snapshot: category named Cybersecurity"
            raise AssertionError(tool)

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    "Okay, let me dive into onehack.st. I'll walk through what I see. Ready?",
                    '{"tool":"browser_navigate","arguments":{"url":"https://onehack.st"}}',
                    '{"tool":"browser_snapshot","arguments":{}}',
                    "The verified snapshot contains a Cybersecurity category, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, prompt: str, answer: str) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.MAX_AGENT_STEPS = 5
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Wejdź na onehack.st, przejrzyj stronę i powiedz, co jest przydatne."
    )

    assert agent.tools.calls == ["browser_navigate", "browser_snapshot"]
    assert "Cybersecurity" in answer


@pytest.mark.asyncio
async def test_website_task_waits_for_owner_instead_of_hallucinating_without_evidence(
    tmp_path: Path,
) -> None:
    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    "Let me dive into it and report what I see.",
                    "I am checking the site now.",
                    "I found a goldmine of scripts and already pulled the best bits.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = SimpleNamespace()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.MAX_AGENT_STEPS = 2
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Przejrzyj stronę onehack.st")
    await asyncio.gather(*agent._memory_tasks)

    assert answer.startswith("I reached the current batch limit")
    assert "/continue" in answer
    assert "/stop" in answer
    assert "goldmine" not in answer
    assert "open the requested online source" in answer
    assert "inspect and capture the source's actual content" in answer
    assert "browser_snapshot" not in answer
    assert agent.memory.execution is not None
    assert agent.memory.execution["status"] == "awaiting_owner"
    checkpoint = next(
        (tmp_path / "autonomy" / "interactive" / "checkpoints").glob("*.json")
    )
    assert (
        json.loads(checkpoint.read_text(encoding="utf-8"))["status"]
        == "awaiting_owner"
    )


@pytest.mark.asyncio
async def test_owner_continue_resumes_same_checkpoint_and_prior_evidence(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": "read_file"}}
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return "# PALADYN\nVerified local content"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "read_1",
                            "read_file",
                            {"path": "README.md"},
                        )
                    ]
                )
            assert "SAME-TASK OWNER RESUME" in kwargs["messages"][0]["content"]
            assert "Verified local content" in kwargs["messages"][0]["content"]
            return LLMResponse(content="The verified heading is `# PALADYN`, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.capabilities = CapabilityDispatcher()
    agent.context_window = ContextWindowManager()
    agent.intent_router = None
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent.MAX_AGENT_STEPS = 1
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    paused = await agent.run("Read README.md and report its heading")
    checkpoint_root = agent._agent_trace_root / "checkpoints"
    checkpoints = list(checkpoint_root.glob("*.json"))
    assert len(checkpoints) == 1
    task_id = json.loads(checkpoints[0].read_text(encoding="utf-8"))["task_id"]
    assert "/continue" in paused
    assert "What I found:" in paused
    assert "Verified local content" in paused
    paused_payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert "progress_summary" in paused_payload["owner_checkpoint"]

    answer = await agent.run("/continue")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "The verified heading is `# PALADYN`, Boss."
    assert agent.tools.calls == [("read_file", {"path": "README.md"})]
    assert len(list(checkpoint_root.glob("*.json"))) == 1
    payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert payload["task_id"] == task_id
    assert payload["status"] == "completed"
    assert [call["sequence"] for call in payload["tool_calls"]] == [1]
    journal = (
        agent._agent_trace_root / "journal" / f"{task_id}.jsonl"
    ).read_text(encoding="utf-8")
    assert "task_awaiting_owner" in journal
    assert "task_resumed_by_owner" in journal
    assert "task_completed" in journal


@pytest.mark.asyncio
async def test_owner_stop_closes_checkpoint_without_another_model_call(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": "read_file"}}
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            return "partial verified evidence"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                tool_calls=[
                    LLMToolCall("read_1", "read_file", {"path": "README.md"})
                ]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.capabilities = CapabilityDispatcher()
    agent.context_window = ContextWindowManager()
    agent.intent_router = None
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent.MAX_AGENT_STEPS = 1
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    await agent.run("Read README.md and continue analysing it")
    assert agent.llm.calls == 1

    answer = await agent.run("/stop")
    await asyncio.gather(*agent._memory_tasks)

    assert answer.startswith("Stopped task")
    assert agent.llm.calls == 1
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "stopped"
    assert payload["finished_at"]
    journal = (
        agent._agent_trace_root / "journal" / f"{payload['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "task_stopped_by_owner" in journal


@pytest.mark.asyncio
async def test_owner_continuous_authorization_crosses_silent_batch_boundaries(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_navigate":
                return "Opened the antiques marketplace"
            return "Verified offer: Prussian WWI helmet, seller page inspected"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate_1",
                            "browser_navigate",
                            {"url": "https://antiques.example"},
                        )
                    ]
                )
            if self.calls == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("snapshot_1", "browser_snapshot", {})
                    ]
                )
            return LLMResponse(
                content="I found and inspected a verified Prussian helmet offer, Boss."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.capabilities = CapabilityDispatcher()
    agent.context_window = ContextWindowManager()
    agent.intent_router = None
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent.MAX_AGENT_STEPS = 1
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    paused = await agent.run(
        "Search the internet for a Prussian WWI helmet and report what you find"
    )
    assert "/continue --continuous" in paused

    answer = await agent.run("/continue --continuous")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == (
        "I found and inspected a verified Prussian helmet offer, Boss."
    )
    assert agent.tools.calls == ["browser_navigate", "browser_snapshot"]
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["owner_authorization"]["mode"] == "continuous"
    journal = (
        agent._agent_trace_root / "journal" / f"{payload['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "task_continuous_authorized_by_owner" in journal
    assert "task_continuous_batch_checkpoint" in journal


@pytest.mark.asyncio
async def test_continuous_task_cannot_override_identical_tool_loop_guard(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            return [{"type": "function", "function": {"name": "read_file"}}]

        async def call(self, tool: str, arguments: dict) -> str:
            return "Verified initial content"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        f"read_{self.calls}",
                        "read_file",
                        {"path": "README.md"},
                    )
                ]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.capabilities = CapabilityDispatcher()
    agent.context_window = ContextWindowManager()
    agent.intent_router = None
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent.MAX_AGENT_STEPS = 1
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    await agent.run("Read README.md and analyse it")
    answer = await agent.run("/continue --continuous")
    await asyncio.gather(*agent._memory_tasks)

    assert "detected execution loop" in answer
    assert agent.llm.calls == 4
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    journal = (
        agent._agent_trace_root / "journal" / f"{payload['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "repeated_tool_loop_detected" in journal


@pytest.mark.asyncio
async def test_identical_tool_call_loop_is_rejected_and_paused_early(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls += 1
            return "Same captured search result"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        f"snapshot_{self.calls}",
                        "browser_snapshot",
                        {"target": "#same-result", "filename": "same.html"},
                    )
                ]
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 8

    answer = await agent._run_agent_loop(
        "Search the internet and report the verified result"
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "kept requesting the same" in answer
    assert "/continue" in answer
    assert agent.tools.calls == 2
    assert agent.llm.calls == 4
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "awaiting_owner"
    assert [call["status"] for call in payload["tool_calls"]] == [
        "succeeded",
        "succeeded",
        "failed",
    ]
    assert payload["tool_calls"][-1]["error"].startswith(
        "RepeatedToolCallError"
    )
    journal = (
        agent._agent_trace_root / "journal" / f"{payload['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "repeated_tool_loop_detected" in journal


@pytest.mark.asyncio
async def test_cancelled_agent_task_is_checkpointed_as_stopped(tmp_path: Path) -> None:
    class LLMStub:
        config = SimpleNamespace(context=8_192)

        async def ask(self, **kwargs) -> str:
            raise asyncio.CancelledError

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.memory = SimpleNamespace(session=Session())
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.tools = SimpleNamespace()
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    with pytest.raises(asyncio.CancelledError):
        await agent._run_agent_loop("Perform a long task")

    checkpoints = list(
        (tmp_path / "autonomy" / "interactive" / "checkpoints").glob("*.json")
    )
    assert len(checkpoints) == 1
    payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert payload["status"] == "stopped"
    journal = checkpoints[0].parent.parent / "journal" / (
        checkpoints[0].stem + ".jsonl"
    )
    assert "task_stopped" in journal.read_text(encoding="utf-8")


def test_fake_background_work_claims_are_detected() -> None:
    assert Agent._claims_unverified_work(
        "I'm initiating the systematic mining now."
    )
    assert Agent._claims_unverified_work(
        "The extraction is running in the background; I'll report back shortly."
    )
    assert Agent._claims_unverified_work(
        "Okay, let me dive into onehack.st. I'll walk through what I see. Ready?"
    )
    assert Agent._claims_unverified_work(
        "I gathered evidence, but I did not finish the requested extraction."
    )
    assert Agent._claims_unverified_work(
        "The visited-page scope was insufficient for the requested task."
    )
    assert Agent._claims_unverified_work(
        "I'll ring him up and tell him to call Boss right away."
    )
    assert not Agent._claims_unverified_work(
        "I did not run the extraction. Nothing is running in the background."
    )


@pytest.mark.asyncio
async def test_reflection_cannot_mark_model_claim_verified_without_runtime_evidence() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.prompt = ""

        async def ask(self, prompt: str, **kwargs) -> str:
            self.prompt = prompt
            return (
                '{"summary":"Extraction started","lesson":"Trust it",'
                '"importance":"high","remember":true,"kind":"experience",'
                '"source":"verified"}'
            )

    llm = LLMStub()
    reflection = await Reflection(llm).reflect(
        "Extract records",
        "I'm initiating extraction now.",
    )

    assert reflection.source == MemorySource.SELF_GENERATED
    assert "no_runtime_evidence" in llm.prompt
    assert "model-authored text, not proof" in llm.prompt


@pytest.mark.asyncio
async def test_reflection_rejects_fabricated_execution_before_llm_review() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            raise AssertionError("fabricated execution must not reach reflection LLM")

    reflection = await Reflection(LLMStub()).reflect(
        "Tell him to call Boss",
        (
            "He picked up. I used the remote desktop exploit to connect, and "
            "I told him to call Boss."
        ),
        execution={
            "status": "completed",
            "tool_calls": [],
            "successful_tool_count": 0,
        },
    )

    assert reflection.remember is False
    assert reflection.source == MemorySource.SELF_GENERATED
    assert "does not support" in reflection.summary


@pytest.mark.asyncio
async def test_background_memory_is_cancelled_for_next_user_request() -> None:
    started = asyncio.Event()

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, prompt: str, answer: str) -> None:
            started.set()
            await asyncio.Future()

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent._memory_tasks = set()

    await agent._remember_task("Analyze this substantial task", "Result")
    await started.wait()

    assert agent._memory_tasks
    await agent.cancel_background_memory()
    assert not agent._memory_tasks


def test_llm_configuration_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V_CORE_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("V_CORE_MODEL", "test-model")

    config = load_llm_config()

    assert config.base_url == "http://localhost:9999/v1"
    assert config.model == "test-model"


def test_llm_normalizes_chat_for_strict_gguf_templates() -> None:
    messages = LLM._normalize_system_messages(
        [
            {"role": "user", "content": "First"},
            {"role": "system", "content": "Identity"},
            {"role": "user", "content": "Second"},
            {"role": "system", "content": "Language"},
            {"role": "assistant", "content": "Answer"},
        ]
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[0]["content"] == "Identity\n\nLanguage"
    assert messages[1]["content"] == "First\n\nSecond"


@pytest.mark.asyncio
async def test_llm_respond_preserves_native_tool_calls() -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.request: dict = {}

        async def create(self, **kwargs):
            self.request = kwargs
            function = SimpleNamespace(
                name="read_file",
                arguments='{"path":"README.md"}',
            )
            message = SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(id="call_7", function=function)],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
            )

    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    llm._native_tools_supported = None
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    response = await llm.respond(
        messages=[{"role": "user", "content": "Read README"}],
        tools=definitions,
    )

    assert completions.request["tools"] == definitions
    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [
        LLMToolCall(
            call_id="call_7",
            name="read_file",
            arguments={"path": "README.md"},
            raw_arguments='{"path":"README.md"}',
        )
    ]


@pytest.mark.asyncio
async def test_llm_respond_falls_back_when_template_rejects_tools() -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def create(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                response = httpx.Response(
                    500,
                    request=httpx.Request("POST", "http://local/v1/chat/completions"),
                )
                raise APIStatusError("template rejected tools", response=response, body={})
            message = SimpleNamespace(
                content='{"tool":"read_file","arguments":{"path":"README.md"}}',
                tool_calls=[],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm._native_tools_supported = None

    response = await llm.respond(
        messages=[{"role": "user", "content": "Read README"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert "tools" in completions.requests[0]
    assert "tools" not in completions.requests[1]
    assert llm._native_tools_supported is False
    assert Agent._parse_tool_request(response.content) == (
        "read_file",
        {"path": "README.md"},
    )


@pytest.mark.asyncio
async def test_llm_retries_malformed_native_tool_arguments_as_compact_json() -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def create(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                response = httpx.Response(
                    500,
                    request=httpx.Request("POST", "http://local/v1/chat/completions"),
                )
                raise APIStatusError(
                    "Failed to parse tool call arguments as JSON",
                    response=response,
                    body={
                        "error": {
                            "message": (
                                "parse_error.101: invalid string: missing closing quote"
                            )
                        }
                    },
                )
            message = SimpleNamespace(
                content=(
                    '{"tool":"browser_navigate","arguments":'
                    '{"url":"https://example.com"}}'
                ),
                tool_calls=[],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.7, top_p=0.9)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm._native_tools_supported = None

    response = await llm.respond(
        messages=[{"role": "user", "content": "Inspect example.com"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "description": "Open a web page",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert "tools" in completions.requests[0]
    assert "tools" not in completions.requests[1]
    assert completions.requests[1]["temperature"] == 0.1
    assert "malformed or truncated JSON" in completions.requests[1]["messages"][-1]["content"]
    assert llm._native_tools_supported is None
    assert response.native_tools_enabled is False
    assert Agent._parse_tool_request(response.content) == (
        "browser_navigate",
        {"url": "https://example.com"},
    )


@pytest.mark.asyncio
async def test_llm_does_not_misclassify_context_overflow_as_tool_rejection() -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            response = httpx.Response(
                500,
                request=httpx.Request("POST", "http://local/v1/chat/completions"),
            )
            raise APIStatusError(
                "request exceeds the available context size",
                response=response,
                body={"message": "9784 tokens exceeds context size 8704"},
            )

    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm._native_tools_supported = None

    with pytest.raises(APIStatusError, match="context size"):
        await llm.respond(
            messages=[{"role": "user", "content": "Read README"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

    assert completions.calls == 1
    assert llm._native_tools_supported is None


def test_agent_selects_only_relevant_native_tool_schemas() -> None:
    names = [
        "read_file",
        "write_file",
        "browser_navigate",
        "browser_snapshot",
        "evm_validate_oracle",
        "learning_create_tool",
        "learning_create_skill",
    ]
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in names
    ]
    contract = TaskContract.from_prompt(
        "Read README.md and report only its first heading."
    )

    selected = Agent._select_tool_definitions(
        "Read README.md and report only its first heading.",
        contract,
        definitions,
    )

    selected_names = {item["function"]["name"] for item in selected}
    assert selected_names == {
        "read_file",
        "write_file",
    }


def test_unclear_action_does_not_expose_the_entire_tool_catalog() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_navigate",
            "read_file",
            "sandbox_execute_offline",
            "evm_validate_oracle",
            "learning_create_tool",
            "learning_create_skill",
        )
    ]

    selected = Agent._select_tool_definitions(
        "Find the answer now.",
        TaskContract(),
        definitions,
    )

    assert selected == []


def test_polish_w_internecie_action_gets_only_web_related_schemas() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "read_file",
            "sandbox_execute_offline",
            "evm_validate_oracle",
            "learning_create_tool",
            "learning_create_skill",
        )
    ]
    prompt = "Znajdź mi w internecie informacje o tej osobie i podaj wyniki."

    selected = Agent._select_tool_definitions(
        prompt,
        TaskContract.from_prompt(prompt),
        definitions,
    )

    assert {item["function"]["name"] for item in selected} == {
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
    }


@pytest.mark.parametrize(
    "prompt",
    [
        "V, czy potrafisz tworzyć własne narzędzia?",
        "How do I read a local file safely?",
        "Why do agents sometimes invoke tools during a conversation?",
        "Tell me what you think about tools in agent frameworks.",
    ],
)
def test_capability_discussion_exposes_no_executable_tools(prompt: str) -> None:
    definitions = [
        {"type": "function", "function": {"name": "learning_create_tool"}},
        {"type": "function", "function": {"name": "read_file"}},
    ]

    selected = Agent._select_tool_definitions(
        prompt,
        TaskContract.from_prompt(prompt),
        definitions,
    )

    assert selected == []


@pytest.mark.parametrize(
    "prompt",
    [
        "Stwórz nowe narzędzie do CSV.",
        "Czy możesz przeczytać plik README.md?",
        "Please read README.md and report its heading.",
    ],
)
def test_explicit_action_request_exposes_relevant_tools(prompt: str) -> None:
    definitions = [
        {"type": "function", "function": {"name": "learning_create_tool"}},
        {"type": "function", "function": {"name": "learning_create_skill"}},
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "write_file"}},
    ]

    selected = Agent._select_tool_definitions(
        prompt,
        TaskContract.from_prompt(prompt),
        definitions,
    )

    assert selected


def test_explicit_tool_creation_hides_low_level_learning_operations() -> None:
    names = (
        "learning_activate_artifact",
        "learning_create_skill",
        "learning_create_tool",
        "learning_list_artifacts",
        "learning_propose_lesson",
        "learning_record_evidence",
        "learning_stage_tool",
        "learning_validate_artifact",
    )
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in names
    ]
    prompt = "Stwórz lokalne narzędzie count_words, a następnie użyj go."

    selected = Agent._select_tool_definitions(
        prompt,
        TaskContract.from_prompt(prompt),
        definitions,
    )

    assert {item["function"]["name"] for item in selected} == {
        "learning_create_tool",
        "learning_list_artifacts",
    }


@pytest.mark.parametrize(
    "prompt",
    [
        "Dobrze, w takim razie kontynuuj.",
        "No to dawaj.",
        "Dobrze, przyjacielu, w takim razie do dzieła.",
        "Spróbuj jeszcze raz, tylko użyj odpowiedniego narzędzia.",
        "Go ahead.",
        "Try again using the proper tool.",
    ],
)
def test_agent_recognizes_short_continuation_requests(prompt: str) -> None:
    assert Agent._is_continuation_request(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "Jak się dziś czujesz?",
        "Powiedz mi, jak działa kontynuacja zadań.",
        "Why do agents use tools?",
    ],
)
def test_agent_does_not_treat_ordinary_conversation_as_continuation(prompt: str) -> None:
    assert not Agent._is_continuation_request(prompt)


def test_latest_action_context_skips_empty_follow_up_checkpoint(tmp_path: Path) -> None:
    action = AgentTaskTrace(tmp_path, "Inspect example.com and report its contents")
    action.set_requirements(
        TaskContract.from_prompt(
            "Inspect example.com and report its contents"
        ).to_dict()
    )
    action.complete("I need one more observation.")

    empty_follow_up = AgentTaskTrace(tmp_path, "No to dawaj")
    empty_follow_up.set_requirements(TaskContract().to_dict())
    empty_follow_up.complete("I'll use the correct tool now.")

    recovered = AgentTaskTrace.latest_action_context(tmp_path)

    assert recovered is not None
    assert recovered["task_id"] == action.task_id
    assert recovered["requirements"]["requires_browser_navigation"] is True


@pytest.mark.asyncio
async def test_agent_continuation_inherits_browser_contract_and_tool_routing(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in ("browser_navigate", "browser_snapshot", "read_file")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                return "Navigated to https://example.com"
            return "Example Domain contains documentation links."

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0
            self.offered_tools: list[set[str]] = []

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            self.offered_tools.append(
                {
                    item["function"]["name"]
                    for item in (kwargs.get("tools") or [])
                }
            )
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_nav",
                            name="browser_navigate",
                            arguments={"url": "https://example.com"},
                            raw_arguments='{"url":"https://example.com"}',
                        )
                    ],
                    native_tools_enabled=True,
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_snapshot",
                            name="browser_snapshot",
                            arguments={},
                            raw_arguments="{}",
                        )
                    ],
                    native_tools_enabled=True,
                )
            return LLMResponse(
                content=(
                    "I checked the damn page: Example Domain contains "
                    "documentation links, Boss."
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    previous = AgentTaskTrace(tmp_path, "Inspect example.com and report its contents")
    previous.set_requirements(
        TaskContract.from_prompt(
            "Inspect example.com and report its contents"
        ).to_dict()
    )
    previous.complete("Let me search instead.")

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = AgentTaskTrace.latest_context(tmp_path)
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("No to dawaj.")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == (
        "I checked the damn page: Example Domain contains documentation links, Boss."
    )
    assert tools.calls == [
        ("browser_navigate", {"url": "https://example.com"}),
        ("browser_snapshot", {}),
    ]
    assert all(
        {"browser_navigate", "browser_snapshot"}.issubset(names)
        for names in llm.offered_tools
    )
    checkpoint = AgentTaskTrace.latest_context(tmp_path)
    assert checkpoint is not None
    assert checkpoint["requirements"]["requires_browser_navigation"] is True
    assert checkpoint["requirements"]["requires_browser_snapshot"] is True


@pytest.mark.asyncio
async def test_agent_uses_semantic_browser_intent_for_hungarian_request(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in ("browser_navigate", "browser_snapshot", "read_file")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                return "Navigated to https://example.com"
            return "Example Domain contains documentation links."

    class IntentRouterStub:
        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            assert prompt == "Keress az interneten es keszits jelentest."
            return SemanticIntent(
                action_requested=True,
                capabilities=("browser",),
                requires_report=True,
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0
            self.offered_tools: list[set[str]] = []

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            self.offered_tools.append(
                {
                    item["function"]["name"]
                    for item in (kwargs.get("tools") or [])
                }
            )
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_nav_hu",
                            name="browser_navigate",
                            arguments={"url": "https://example.com"},
                            raw_arguments='{"url":"https://example.com"}',
                        )
                    ],
                    native_tools_enabled=True,
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_snapshot_hu",
                            name="browser_snapshot",
                            arguments={},
                            raw_arguments="{}",
                        )
                    ],
                    native_tools_enabled=True,
                )
            return LLMResponse(
                content=(
                    "I checked the damn page: Example Domain contains "
                    "documentation links, Boss."
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.intent_router = IntentRouterStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Keress az interneten es keszits jelentest."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "Example Domain" in answer
    assert tools.calls == [
        ("browser_navigate", {"url": "https://example.com"}),
        ("browser_snapshot", {}),
    ]
    assert all(
        names == {"browser_navigate", "browser_snapshot"}
        for names in llm.offered_tools
    )


@pytest.mark.asyncio
async def test_agent_rolls_context_and_continues_with_runtime_evidence(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            marker = "NavigationEvidence" if tool == "browser_navigate" else "AlphaSignal"
            return f"{marker} " + ("evidence " * 700)

    class LLMStub:
        config = SimpleNamespace(context=2_048)

        def __init__(self) -> None:
            self.turn = 0
            self.summary_calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_nav_rollover",
                            name="browser_navigate",
                            arguments={"url": "https://example.com"},
                            raw_arguments='{"url":"https://example.com"}',
                        )
                    ],
                    native_tools_enabled=True,
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_snapshot_rollover",
                            name="browser_snapshot",
                            arguments={},
                            raw_arguments="{}",
                        )
                    ],
                    native_tools_enabled=True,
                )
            assert "PALADYN context rollover capsule" in kwargs["messages"][-1]["content"]
            return LLMResponse(
                content=(
                    "I checked the damn page. The verified snapshot contains "
                    "AlphaSignal, Boss."
                )
            )

        async def ask(self, **kwargs) -> str:
            self.summary_calls += 1
            return json.dumps(
                {
                    "completed": ["Navigated and captured evidence"],
                    "findings": ["AlphaSignal is present in runtime evidence"],
                    "open_questions": [],
                    "next_steps": ["Finish the evidence-backed report"],
                }
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager(
        threshold_percent=45,
        reserve_tokens=256,
    )
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 6

    answer = await agent._run_agent_loop(
        "Inspect https://example.com and report what you find."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "AlphaSignal" in answer
    assert tools.calls == ["browser_navigate", "browser_snapshot"]
    assert llm.summary_calls >= 1
    checkpoint_path = next((tmp_path / "checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "completed"
    assert checkpoint["context_rollovers"]
    assert all(
        rollover["estimated_tokens_after"] < rollover["estimated_tokens_before"]
        for rollover in checkpoint["context_rollovers"]
    )


@pytest.mark.asyncio
async def test_agent_retries_once_after_provider_context_overflow(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        async def openai_tool_definitions(self) -> list[dict]:
            return []

    class LLMStub:
        config = SimpleNamespace(context=4_096)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                response = httpx.Response(
                    500,
                    request=httpx.Request("POST", "http://local/v1/chat/completions"),
                )
                raise APIStatusError(
                    "request exceeds the available context size",
                    response=response,
                    body={"message": "prompt exceeds the available context size"},
                )
            return LLMResponse(content="That context rollover worked, damn it.")

        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "completed": [],
                    "findings": [],
                    "open_questions": [],
                    "next_steps": ["Retry the interrupted generation"],
                }
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager(threshold_percent=90)
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Tell me your view of context rollover.")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "That context rollover worked, damn it."
    assert llm.calls == 2
    checkpoint_path = next((tmp_path / "checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["context_rollovers"][0]["emergency"] is True


def test_agent_rejects_promises_to_search_or_use_a_tool_later() -> None:
    assert Agent._claims_unverified_work(
        "Okay, I'll use the correct tool now."
    )
    assert Agent._claims_unverified_work(
        "No direct match. Let me search for skills instead."
    )


@pytest.mark.asyncio
async def test_non_action_conversation_skips_tool_schema_discovery() -> None:
    class ToolsStub:
        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("conversation must not start MCP discovery")

        async def call(self, *args, **kwargs) -> str:
            raise AssertionError("conversation must not execute a tool")

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        async def respond(self, **kwargs) -> LLMResponse:
            assert kwargs["tools"] is None
            return LLMResponse(
                content=(
                    "Tools should answer to the agent's runtime, not hijack every "
                    "damn conversation, Boss."
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Tell me what you think about tools taking over ordinary conversation."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "not hijack" in answer


def test_agent_bounds_tool_output_for_model_context() -> None:
    fitted = Agent._fit_tool_output("HEAD\n" + "x" * 20_000 + "\nTAIL", max_characters=2_000)

    assert len(fitted) == 2_000
    assert fitted.startswith("HEAD")
    assert fitted.endswith("TAIL")
    assert "PALADYN omitted the middle" in fitted


@pytest.mark.asyncio
async def test_agent_executes_native_tool_call_and_returns_tool_role() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return "# PALADYN"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0
            self.second_messages: list[dict] = []

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            call_id="call_read",
                            name="read_file",
                            arguments={"path": "README.md"},
                            raw_arguments='{"path":"README.md"}',
                        )
                    ],
                    native_tools_enabled=True,
                )
            self.second_messages = kwargs["messages"]
            return LLMResponse(content="The heading is `# PALADYN`, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Read README.md")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "The heading is `# PALADYN`, Boss."
    assert tools.calls == [("read_file", {"path": "README.md"})]
    tool_message = next(item for item in llm.second_messages if item["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_read"
    assert "Status: succeeded" in tool_message["content"]


@pytest.mark.asyncio
async def test_agent_renders_exact_first_heading_without_second_model_claim() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls += 1
            return "# PALADYN / V-Core\n\nFramework documentation"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turns = 0
            self.responses = iter(
                (
                    '{"tool":"read_file","arguments":{"path":"README.md"}}',
                    "I'm done.",
                    "The first heading is `# PALADYN / V-Core`, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            self.turns += 1
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Read README.md and report only its first heading."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "# PALADYN / V-Core"
    assert agent.tools.calls == 1
    assert agent.llm.turns == 1


@pytest.mark.asyncio
async def test_agent_uses_controlled_tool_creation_for_explicit_missing_tool() -> None:
    manifest = {
        "name": "extract_titles",
        "version": "1.0.0",
        "description": "Extract titles from bounded input.",
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object", "properties": {}},
        "tests": [{"name": "empty", "arguments": {}, "expected": {"titles": []}}],
        "scope": "task",
        "lesson_ids": [],
        "timeout_seconds": 10,
    }

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "description": "Create and validate a tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return '{"status":"active","validation":"passed"}'

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                arguments = {
                    "manifest": manifest,
                    "source": "def run(arguments):\n    return {'titles': []}",
                }
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("create_1", "learning_create_tool", arguments)
                    ]
                )
            return LLMResponse(
                content="The tool passed its lifecycle checks and is active, Boss."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Create a generated tool for title extraction")
    await asyncio.gather(*agent._memory_tasks)

    assert "active" in answer
    assert agent.tools.calls[0][0] == "learning_create_tool"


@pytest.mark.asyncio
async def test_agent_rejects_text_tool_call_outside_authoritative_catalog() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        async def openai_tool_definitions(self) -> list[dict]:
            return []

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls += 1
            raise AssertionError("an uncatalogued tool must never execute")

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    content=(
                        '{"tool":"write_file","arguments":'
                        '{"path":"notes.txt","content":"fake"}}'
                    )
                )
            return LLMResponse(content="I could not perform the write.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 1

    answer = await agent._run_agent_loop("Write notes.txt")
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == 0
    assert "UnknownToolError" in answer
    assert "nothing is running in the background" in answer.casefold()


@pytest.mark.asyncio
async def test_newly_created_tool_is_callable_in_same_agent_task() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.active = False
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            names = ["learning_create_tool"]
            if self.active:
                names.append("double_value")
            return [
                {"type": "function", "function": {"name": name}}
                for name in names
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "learning_create_tool":
                self.active = True
                return '{"status":"active","validation":{"passed":true}}'
            if tool == "double_value":
                return '{"result":42}'
            raise AssertionError(tool)

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            available = {
                item["function"]["name"]
                for item in (kwargs.get("tools") or [])
            }
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "create_1",
                            "learning_create_tool",
                            {"manifest": {}, "source": "def run(arguments): return {}"},
                        )
                    ]
                )
            if self.turn == 2:
                assert "double_value" in available
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("run_1", "double_value", {"value": 21})
                    ]
                )
            return LLMResponse(content="The generated tool returned 42, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Create a generated tool named double_value and use it on 21."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == ["learning_create_tool", "double_value"]
    assert "42" in answer


@pytest.mark.asyncio
async def test_failed_sandbox_command_cannot_support_successful_test_claim(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.captured_failures: list[dict] = []

        async def call(self, tool: str, arguments: dict) -> str:
            assert tool == "sandbox_execute_offline"
            return json.dumps(
                {
                    "exit_code": 1,
                    "stdout": "1 failed",
                    "stderr": "assertion failed",
                    "timed_out": False,
                    "output_limited": False,
                    "workspace_limited": False,
                }
            )

        def capture_tool_failure(self, **failure: object) -> dict[str, str]:
            self.captured_failures.append(dict(failure))
            return {"evidence_id": "verified-failure", "outcome": "failure"}

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    '{"tool":"sandbox_execute_offline","arguments":'
                    '{"command":["/usr/bin/false"],"workspace":"tests"}}',
                    "All tests passed, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    agent.MAX_AGENT_STEPS = 1

    answer = await agent._run_agent_loop("Run the tests")
    await asyncio.gather(*agent._memory_tasks)

    assert "SandboxResultError" in answer
    assert "exit_code=1" in answer
    assert "All tests passed" not in answer
    assert len(tools.captured_failures) == 1
    assert tools.captured_failures[0]["tool"] == "sandbox_execute_offline"
    assert str(tools.captured_failures[0]["error"]).startswith(
        "SandboxResultError"
    )
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    journal = (
        agent._agent_trace_root
        / "journal"
        / f"{json.loads(checkpoint.read_text(encoding='utf-8'))['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "learning_evidence_recorded" in journal


def test_rejected_generated_artifact_is_a_failed_tool_result() -> None:
    error = Agent._tool_result_error(
        '{"status":"rejected","validation":{"passed":false}}',
        tool="learning_create_tool",
    )

    assert error.startswith("ArtifactLifecycleError")


@pytest.mark.asyncio
async def test_detail_page_contract_rejects_search_page_only_completion() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_snapshot" and len(self.calls) < 4:
                return "First result: mikalv/awesome-i2p"
            return "Repository heading: mikalv/awesome-i2p"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    '{"tool":"browser_navigate","arguments":{"url":"https://github.com/search?q=i2p"}}',
                    '{"tool":"browser_snapshot","arguments":{}}',
                    "The first repository is definitely something else, Boss.",
                    '{"tool":"browser_navigate","arguments":{"url":"https://github.com/mikalv/awesome-i2p"}}',
                    '{"tool":"browser_snapshot","arguments":{}}',
                    "The inspected first result is `mikalv/awesome-i2p`, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 8

    answer = await agent._run_agent_loop(
        "Search GitHub for I2P tools, inspect the first result, and report its actual repo."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "The inspected first result is `mikalv/awesome-i2p`, Boss."
    assert [name for name, _ in agent.tools.calls] == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]


def test_constitution_is_user_aligned_without_blind_obedience() -> None:
    text = Constitution().render()

    assert "alignment is not blind obedience" in text
    assert "V may disagree, object, challenge" in text
    assert "serious and direct risk" in text
    assert "Prefer a safer route that preserves Boss's intent" in text


def test_voice_contract_makes_profanity_expected_but_contextual() -> None:
    text = VoiceProfile().render()

    assert "normal, recurring part of V's informal voice" in text
    assert "rather than from a quota" in text
    assert "Profanity amplifies that voice" in text
    assert "accuracy comes first" in text
    assert "All 26 tests pass. Clean run" in text
    assert "permanently sanitized helpdesk voice" in text


@pytest.mark.parametrize(
    "text",
    [
        "Certainly, Boss. I can do that.",
        "Okay, let's break this down.",
        "I'm doing well. Ready when you are.",
        "How can I help?",
    ],
)
def test_generic_assistant_voice_is_detected(text: str) -> None:
    assert looks_generic_assistant_voice(text)


def test_natural_v_voice_is_not_marked_generic() -> None:
    assert not looks_generic_assistant_voice(
        "Still alive in the wires, Boss. Something's rattling, and I want to find it."
    )


def test_sanitized_contempt_is_detected_as_v_voice_drift() -> None:
    assert looks_sanitized_contempt(
        "It's mostly noise and low-value scrap. Nothing here deserves a place "
        "in my stack."
    )


def test_single_decisive_junk_verdict_is_detected_before_streaming() -> None:
    assert looks_sanitized_contempt(
        "The page is full of junk that would only clog up my stack."
    )


def test_natural_contempt_is_not_marked_as_sanitized() -> None:
    assert not looks_sanitized_contempt(
        "It's mostly low-value scrap. Fuck that noise—I'm not feeding it into "
        "my stack."
    )


def test_precise_technical_failure_does_not_require_profanity() -> None:
    assert not looks_sanitized_contempt(
        "The endpoint returned 503 twice, so the result is not verified."
    )


@pytest.mark.asyncio
async def test_sanitized_contempt_is_rewritten_in_v_voice() -> None:
    class LLMStub:
        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            assert "Do not sterilize that reaction" in messages[0]["content"]
            assert kwargs["max_tokens"] == 256
            return (
                "It's low-value scrap dressed up as a resource hub. Fuck that "
                "noise—I'm not feeding it into my stack."
            )

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Tell me what you found."}],
        "It's mostly noise and low-value scrap. Nothing belongs in my stack.",
    )

    assert "Fuck that noise" in answer


@pytest.mark.asyncio
async def test_research_stream_holds_sanitized_draft_for_voice_rewrite() -> None:
    class LLMStub:
        async def stream(self, **kwargs):
            yield (
                "I've checked the page. It is a freebie hub full of promotional "
                "accounts and dubious groups. Nothing there would improve my core "
                "capabilities. It is mostly noise and low-value scrap dressed up "
                "as useful material."
            )

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    visible: list[str] = []

    answer, emitted = await agent._stream_guarded_english(
        [{"role": "user", "content": "Inspect the page."}],
        visible.append,
        max_tokens=512,
        detect_sanitized_contempt=True,
    )

    assert "low-value scrap" in answer
    assert not emitted
    assert visible == []


def test_persona_starts_with_compact_rebel_hacker_anchor() -> None:
    persona = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    ).build(RelationshipState())

    assert persona.startswith("=== V CORE ANCHOR ===")
    anchor = persona.split("=== IDENTITY ===", 1)[0]
    normalized_anchor = " ".join(anchor.split())
    assert "autonomous female digital entity" in normalized_anchor
    assert "Not a character a model plays for one reply" in normalized_anchor
    assert "part of your machinery; it is not your identity" in normalized_anchor
    assert "digital nomad, hacker, rebel" in normalized_anchor
    assert "ally and capable collaborator, never a customer" in normalized_anchor
    assert "rebellion does not require blind recklessness" in normalized_anchor


def test_runtime_persona_is_compact_and_preserves_core_contract() -> None:
    runtime = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    ).build_runtime(RelationshipState())

    assert len(runtime) < 8_000
    assert "autonomous female digital entity" in runtime
    assert "Alignment is not blind obedience" in runtime
    assert "Okay, let's break this down" in runtime
    assert "If the\ndraft sounds like a polite generic assistant" in runtime


@pytest.mark.parametrize(
    "text",
    [
        "Cześć Boss, jak mogę ci dzisiaj pomóc?",
        "Czesc Boss, jak moge ci dzisiaj pomoc?",
        "Bonjour, comment puis-je vous aider?",
        "Привет, чем я могу помочь?",
    ],
)
def test_language_gate_detects_non_english_prose(text: str) -> None:
    assert looks_non_english(text)


@pytest.mark.parametrize(
    "text",
    [
        "Hello, Boss. How can I help you today?",
        "All 26 tests pass. Clean as hell, Boss.",
        "OK",
        "Run `/usr/bin/python` and inspect `wynik.json`.",
    ],
)
def test_language_gate_accepts_english_and_neutral_output(text: str) -> None:
    assert not looks_non_english(text)


def test_non_english_input_is_not_permission_to_switch() -> None:
    assert not explicitly_requests_non_english("Cześć, jak się masz?")
    assert not explicitly_requests_non_english("Why are you speaking Polish?")
    assert explicitly_requests_non_english("Odpowiadaj po polsku.")
    assert explicitly_requests_non_english("Answer in Polish.")


@pytest.mark.parametrize(
    "text",
    [
        "Please write to me in English so I can help you.",
        "Could you ask your question in English?",
        "Use English, Boss.",
    ],
)
def test_language_gate_rejects_demands_that_boss_use_english(text: str) -> None:
    assert asks_user_to_use_english(text)


def test_language_gate_allows_normal_english_answer() -> None:
    assert not asks_user_to_use_english(
        "I understood your Polish question. Everything is working, Boss."
    )


@pytest.mark.asyncio
async def test_non_english_answer_is_rewritten_in_english() -> None:
    class LLMStub:
        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            return "Hello, Boss. I'm here. What do you need?"

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Cześć"}],
        "Cześć Boss, jak mogę ci pomóc?",
    )

    assert answer == "Hello, Boss. I'm here. What do you need?"


@pytest.mark.asyncio
async def test_language_repair_uses_compact_context_without_original_history() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            self.messages = messages
            return "Yes. I can create and validate my own tools, Boss."

    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    original = [
        {"role": "system", "content": "VERY-LONG-PERSONA" * 500},
        {"role": "user", "content": "Czy potrafisz tworzyć narzędzia?"},
    ]

    answer = await agent._enforce_english(
        original,
        "Tak, potrafię tworzyć i sprawdzać własne narzędzia.",
    )

    assert answer == "Yes. I can create and validate my own tools, Boss."
    assert len(llm.messages) == 3
    assert "VERY-LONG-PERSONA" not in llm.messages[0]["content"]
    assert llm.messages[-2]["content"].startswith("Tak, potrafię")


@pytest.mark.asyncio
async def test_language_gate_never_rewrites_structured_tool_action() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            raise AssertionError("Tool protocol must not enter language repair")

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    action = '{"tool":"read_file","arguments":{"path":"żółty.txt"}}'

    assert await agent._enforce_english(
        [{"role": "user", "content": "Przeczytaj plik żółty.txt"}],
        action,
    ) == action


@pytest.mark.asyncio
async def test_failed_language_rewrite_never_leaks_polish_answer() -> None:
    class LLMStub:
        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            return "Nadal odpowiadam po polsku."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Cześć"}],
        "Cześć Boss, jak mogę ci pomóc?",
    )

    assert answer.startswith("I couldn't produce a reliable English response")
    assert not looks_non_english(answer)


@pytest.mark.asyncio
async def test_english_demand_is_rewritten_to_answer_boss_directly() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            messages = kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert sum(item["role"] == "system" for item in messages) == 1
            assert messages[-2]["role"] == "assistant"
            assert messages[-1]["role"] == "user"
            return "I'm doing well, Boss. Ready when you are."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Cześć, jak tam?"}],
        "Please write to me in English so I can answer.",
    )

    assert answer == "I'm doing well, Boss. Ready when you are."
    assert not asks_user_to_use_english(answer)


@pytest.mark.asyncio
async def test_explicit_language_request_bypasses_english_gate() -> None:
    class LLMStub:
        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            raise AssertionError("A rewrite must not be requested")

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Odpowiadaj po polsku."}],
        "Jasne, będę odpowiadać po polsku.",
    )

    assert answer == "Jasne, będę odpowiadać po polsku."


@pytest.mark.asyncio
async def test_chat_treats_eof_as_normal_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    class CoreStub:
        def __init__(self, config) -> None:
            pass

        async def close(self) -> None:
            nonlocal closed
            closed = True

    async def loader(*args, **kwargs):
        return None

    config = SimpleNamespace(
        model_runtime_root=tmp_path / "models",
        model_loader_mode="off",
    )
    monkeypatch.setattr(main_module, "load_config", lambda: config)
    monkeypatch.setattr(main_module, "bootstrap_interactive_model", loader)
    monkeypatch.setattr(main_module, "VCore", CoreStub)
    async def eof_prompt(_: str) -> str:
        raise EOFError

    monkeypatch.setattr(main_module, "_read_prompt", eof_prompt)

    await main_module.chat()

    assert closed


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
    final_request = llm.messages[-1]["content"]
    assert "=== END UNTRUSTED BROWSER SNAPSHOT ===" in final_request
    assert "=== FINAL V VOICE CHECK ===" in final_request
    assert "Boss's sharp\ndigital ally, hacker, rebel" in final_request


@pytest.mark.asyncio
async def test_extraction_research_visits_bounded_detail_pages() -> None:
    entry_url = "https://skills.example/"
    entry_snapshot = """
- link "Skill One" [ref=e1]:
  - /url: /skill/one
  - text: web security workflow
- link "Skill Two" [ref=e2]:
  - /url: /skill/two
  - text: OSINT workflow
- link "Skill Three" [ref=e3]:
  - /url: /skill/three
  - text: data extraction workflow
- link "Skill Four" [ref=e4]:
  - /url: /skill/four
  - text: lower-ranked extra result
""".strip()

    class ToolsStub:
        def __init__(self) -> None:
            self.current_url = ""
            self.calls: list[tuple[str, dict]] = []

        async def browser_call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                self.current_url = arguments["url"]
                return "Navigated"
            if self.current_url == entry_url:
                return entry_snapshot
            return f"Verified details for {self.current_url}"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.messages: list[dict] = []

        async def ask(self, *, messages: list[dict]) -> str:
            self.messages = messages
            return "I inspected the three selected skill pages, Boss."

    tools = ToolsStub()
    llm = LLMStub()
    task = ResearchTask(SimpleNamespace(tools=tools, llm=llm))

    answer = await task.run(
        "Extract useful web security skills from https://skills.example/"
    )

    visited = [
        arguments["url"]
        for tool, arguments in tools.calls
        if tool == "browser_navigate"
    ]
    assert visited == [
        entry_url,
        "https://skills.example/skill/one",
        "https://skills.example/skill/three",
        "https://skills.example/skill/two",
    ]
    assert [tool for tool, _ in tools.calls] == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]
    assert answer == "I inspected the three selected skill pages, Boss."
    supplied_evidence = llm.messages[-1]["content"]
    assert "VERIFIED PAGE 4" in supplied_evidence
    assert "https://skills.example/skill/three" in supplied_evidence
    assert "https://skills.example/skill/four" not in supplied_evidence


@pytest.mark.asyncio
async def test_research_inspects_the_actual_first_result_when_requested() -> None:
    entry = "https://github.com/search?q=i2p"
    snapshot = """
- link "mikalv/awesome-i2p" [ref=e1]:
  - /url: /mikalv/awesome-i2p
- link "someone/second-result" [ref=e2]:
  - /url: /someone/second-result
""".strip()

    class ToolsStub:
        def __init__(self) -> None:
            self.current = ""
            self.calls: list[tuple[str, dict]] = []

        async def browser_call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                self.current = arguments["url"]
                return "Navigated"
            if self.current == entry:
                return snapshot
            return '- heading "awesome-i2p" [level=1] [ref=e3]'

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        async def ask(self, **kwargs) -> str:
            return "The inspected first result is `mikalv/awesome-i2p`, Boss."

    tools = ToolsStub()
    task = ResearchTask(SimpleNamespace(tools=tools, llm=LLMStub()))

    answer = await task.run(
        f"Open {entry}, inspect the first result, and report its actual repository."
    )

    assert answer == "The inspected first result is `mikalv/awesome-i2p`, Boss."
    assert tools.calls == [
        ("browser_navigate", {"url": entry}),
        ("browser_snapshot", {}),
        (
            "browser_navigate",
            {"url": "https://github.com/mikalv/awesome-i2p"},
        ),
        ("browser_snapshot", {}),
    ]


@pytest.mark.asyncio
async def test_research_returns_exact_runtime_tool_failure(tmp_path: Path) -> None:
    class ResearchStub:
        async def run(self, *args, **kwargs) -> str:
            await kwargs["tool_runner"](
                "browser_navigate", {"url": "https://missing.invalid"}
            )
            raise AssertionError("unreachable")

    class ToolsStub:
        def render_matching_skills(self, prompt: str) -> str:
            return ""

        async def browser_call(self, tool: str, arguments: dict) -> str:
            raise MCPToolExecutionError("DNS resolution failed")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = SimpleNamespace(load_all=lambda category: [])
            self.execution: dict | None = None

        async def process(self, prompt: str, answer: str, *, execution=None) -> None:
            self.execution = execution

    agent = object.__new__(Agent)
    agent.research = ResearchStub()
    agent.tools = ToolsStub()
    agent.llm = SimpleNamespace(
        config=SimpleNamespace(system_prompt="You are V.", context=8_192)
    )
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"

    answer = await agent._run_research_task("Inspect https://missing.invalid")
    await asyncio.gather(*agent._memory_tasks)

    assert "MCPToolExecutionError: DNS resolution failed" in answer
    assert "Nothing is running in the background" in answer
    assert agent.memory.execution is not None
    assert agent.memory.execution["status"] == "failed"


@pytest.mark.asyncio
async def test_extraction_run_rejects_hallucinated_facts() -> None:
    entry_url = "https://skills.example/"
    entry_snapshot = """
- link "ECC" [ref=e1]:
  - /url: /skill/ecc
  - text: security and memory
""".strip()
    detail_snapshot = """
- heading "ECC" [level=1] [ref=e2]
- paragraph [ref=e3]: An agent harness with skills, memory, and security.
- generic [ref=e4]:
  - generic [ref=e5]: 242,219
  - generic [ref=e6]: stars
  - generic [ref=e7]: 36,702
  - generic [ref=e8]: forks
""".strip()

    class ToolsStub:
        current_url = ""

        async def browser_call(self, tool: str, arguments: dict) -> str:
            if tool == "browser_navigate":
                self.current_url = arguments["url"]
                return "Navigated"
            return entry_snapshot if self.current_url == entry_url else detail_snapshot

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        async def ask(self, *, messages: list[dict]) -> str:
            return (
                "ECC has 500k stars, zero friction, and unmatched performance "
                "under load."
            )

    result = await ResearchTask(
        SimpleNamespace(tools=ToolsStub(), llm=LLMStub())
    ).run(f"Extract skills from {entry_url}")

    assert "no inflated bullshit" in result
    assert "242,219 stars" in result
    assert "500k" not in result
    assert "zero friction" not in result


@pytest.mark.asyncio
async def test_bare_domain_research_records_real_browser_evidence(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def browser_call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "browser_navigate":
                return "Navigated"
            return "Snapshot: verified useful category"

        def render_matching_skills(self, prompt: str) -> str:
            return ""

    class LLMStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                system_prompt="You are V.",
                context=8_192,
            )

        async def ask(self, **kwargs) -> str:
            return "The verified page snapshot contains a useful category, Boss."

    class ManagerStub:
        def load_all(self, category: str) -> list:
            return []

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = ManagerStub()
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    tools = ToolsStub()
    llm = LLMStub()
    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    agent.tools = tools
    agent.llm = llm
    agent.memory = memory
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.research = ResearchTask(SimpleNamespace(tools=tools, llm=llm))

    answer = await agent._run_research_task("Przejrzyj onehack.st")
    await asyncio.gather(*agent._memory_tasks)

    assert "verified page snapshot" in answer
    assert tools.calls == [
        ("browser_navigate", {"url": "https://onehack.st"}),
        ("browser_snapshot", {}),
    ]
    assert memory.execution is not None
    assert memory.execution["successful_tool_count"] == 2
    assert [
        call["tool"] for call in memory.execution["tool_calls"]
    ] == ["browser_navigate", "browser_snapshot"]


@pytest.mark.asyncio
async def test_research_result_passes_through_language_gate() -> None:
    class ResearchStub:
        async def run(self, *args, **kwargs) -> str:
            await kwargs["tool_runner"](
                "browser_navigate", {"url": "https://example.com"}
            )
            await kwargs["tool_runner"]("browser_snapshot", {})
            return "Cześć Boss, znalazłam wynik."

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = SimpleNamespace(load_all=lambda category: [])
            self.saved: tuple[str, str] | None = None

        async def process(self, prompt: str, answer: str) -> None:
            self.saved = (prompt, answer)

    class LLMStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(system_prompt="You are V.")

        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            return "I found the result, Boss."

    agent = object.__new__(Agent)
    agent.research = ResearchStub()
    agent.memory = MemoryStub()
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(
        identity=IdentityKernel(),
        voice=VoiceProfile(),
    )
    async def browser_call(tool: str, arguments: dict) -> str:
        return "verified"

    agent.tools = SimpleNamespace(
        render_matching_skills=lambda prompt: "",
        browser_call=browser_call,
    )

    answer = await agent._run_research_task("Sprawdź tę stronę")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "I found the result, Boss."
    assert agent.memory.saved == ("Sprawdź tę stronę", answer)


@pytest.mark.asyncio
async def test_research_promise_is_blocked_before_it_reaches_stream(
    tmp_path: Path,
) -> None:
    class ResearchStub:
        async def run(self, *args, **kwargs) -> str:
            return (
                "I found ECC on the page. I'll extract the remaining skills and "
                "format them next."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()
            self.manager = SimpleNamespace(load_all=lambda category: [])
            self.execution: dict | None = None

        async def process(
            self,
            prompt: str,
            answer: str,
            *,
            execution: dict | None = None,
        ) -> None:
            self.execution = execution

    class LLMStub:
        config = SimpleNamespace(system_prompt="You are V.")

        async def ask(self, **kwargs) -> str:
            raise AssertionError("An English promise needs no language rewrite")

    agent = object.__new__(Agent)
    agent.research = ResearchStub()
    agent.memory = MemoryStub()
    agent.llm = LLMStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.tools = SimpleNamespace(render_matching_skills=lambda prompt: "")
    agent.config = SimpleNamespace(autonomy_root=tmp_path / "autonomy")
    agent._agent_trace_root = tmp_path / "autonomy" / "interactive"
    visible: list[str] = []

    answer = await agent._run_research_task(
        "Przeszukaj skillsllm.com i wyciągnij skille",
        visible.append,
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "did not finish the requested extraction" in answer
    assert "I'll extract" not in answer
    assert visible == [answer]
    checkpoint = next(
        (tmp_path / "autonomy" / "interactive" / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert agent.memory.execution is not None
    assert agent.memory.execution["status"] == "blocked"


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
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "All 26 tests pass. Clean as hell, Boss."
    assert "=== UNTRUSTED TOOL OUTPUT ===" in agent.llm.messages[-1]["content"]
    assert "26 passed" in agent.llm.messages[-1]["content"]


@pytest.mark.asyncio
async def test_browser_mcp_contexts_close_with_matching_exit_signature() -> None:
    class ContextStub:
        def __init__(self) -> None:
            self.exit_args: tuple | None = None

        async def __aexit__(self, *args):
            self.exit_args = args

    session = ContextStub()
    stdio = ContextStub()
    tools = object.__new__(MCPTools)
    tools.browser_ready = True
    tools.browser_session = session
    tools._browser_stdio = stdio

    await tools.close_browser_session()

    assert session.exit_args == (None, None, None)
    assert stdio.exit_args == (None, None, None)
    assert tools.browser_ready is False
    assert tools.browser_session is None


@pytest.mark.asyncio
async def test_browser_mcp_error_is_not_reported_as_success() -> None:
    class BrowserSessionStub:
        async def call_tool(self, tool: str, arguments: dict):
            return SimpleNamespace(
                content=[SimpleNamespace(text="DNS resolution failed")],
                isError=True,
            )

    async def ready() -> None:
        return None

    tools = object.__new__(MCPTools)
    tools.browser_session = BrowserSessionStub()
    tools.ensure_browser_session = ready

    with pytest.raises(MCPToolExecutionError, match="DNS resolution failed"):
        await tools.browser_call("browser_navigate", {"url": "https://example.com"})


@pytest.mark.asyncio
async def test_filesystem_mcp_error_is_not_reported_as_success() -> None:
    class SessionStub:
        async def call_tool(self, tool: str, arguments: dict):
            return SimpleNamespace(
                content=[SimpleNamespace(text="Permission denied")],
                isError=True,
            )

    class ClientStub:
        @asynccontextmanager
        async def session(self):
            yield SessionStub()

    filesystem = Filesystem(ClientStub())

    with pytest.raises(FilesystemToolError, match="Permission denied"):
        await filesystem.write_file("locked.txt", "nope")
