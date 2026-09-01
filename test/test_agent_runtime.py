from __future__ import annotations

import asyncio
import json
import os
import stat
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
    matches_requested_language,
)
from v_core.persona.voice import (
    VoiceProfile,
    looks_bland_clarification,
    looks_empty_action_acknowledgement,
    looks_generic_assistant_voice,
    looks_sanitized_contempt,
    looks_task_offloading,
)
from v_core.relationship import RelationshipState
import v_core.main as main_module


def test_public_version_matches_release() -> None:
    assert v_core.__version__ == "3.7"


def test_tool_request_accepts_structured_json() -> None:
    request = Agent._parse_tool_request(
        '{"tool":"read_file","arguments":{"path":"README.md"}}'
    )

    assert request == (
        "read_file",
        {"path": "README.md"},
    )


def test_tool_request_accepts_exact_hermes_function_call_envelope() -> None:
    assert Agent._parse_tool_request(
        '{"name":"read_file","arguments":{"path":"README.md"}}'
    ) == ("read_file", {"path": "README.md"})


def test_tool_request_accepts_hermes_envelope_with_lifecycle_metadata() -> None:
    assert Agent._parse_tool_request(
        '{"name":"read_file","arguments":{"path":"README.md"},'
        '"version":"","scope":"task","timeout_seconds":120}'
    ) == ("read_file", {"path": "README.md"})


def test_tool_request_rejects_hermes_envelope_with_unknown_metadata() -> None:
    assert Agent._parse_tool_request(
        '{"name":"read_file","arguments":{"path":"README.md"},'
        '"description":"not a call envelope"}'
    ) is None


def test_tool_request_does_not_mistake_artifact_manifest_for_call() -> None:
    assert Agent._parse_tool_request(
        '{"name":"count_words","description":"Count words",'
        '"arguments":{},"source":"def run(arguments): return {}"}'
    ) is None


def test_tool_request_rejects_invalid_arguments() -> None:
    assert Agent._parse_tool_request(
        '{"tool":"read_file","arguments":["README.md"]}'
    ) is None


def test_bare_active_tool_builder_payload_is_recovered() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "learning_create_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "source": {"type": "string"},
                        "test": {"type": "object"},
                    },
                    "required": ["name", "description", "source", "test"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    request = Agent._parse_active_lifecycle_payload(
        '{"name":"count_words","description":"Count words",'
        '"source":"def run(arguments): return {}","test":{}}',
        definitions,
    )

    assert request == (
        "learning_create_tool",
        {
            "name": "count_words",
            "description": "Count words",
            "source": "def run(arguments): return {}",
            "test": {},
        },
    )


def test_bare_payload_recovery_rejects_unknown_fields_and_short_names() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "learning_create_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "source": {"type": "string"},
                        "test": {"type": "object"},
                    },
                    "required": ["name", "description", "source", "test"],
                },
            },
        }
    ]

    assert Agent._parse_active_lifecycle_payload(
        '{"name":"count_words"}', definitions
    ) is None


def test_source_phase_recovers_source_from_one_unclosed_compatibility_wrapper() -> None:
    wrapped = (
        '{"tool":"learning_create_tool","arguments":'
        '{"source":"def run(arguments):\\n    '
        'return {\\"count\\": len(arguments[\\"items\\"])}"}'
    )

    assert Agent._parse_generated_tool_source(wrapped) == (
        'def run(arguments):\n    return {"count": len(arguments["items"])}'
    )


def test_source_phase_prompt_binds_runtime_fixture_fields() -> None:
    prompt = Agent._generated_source_phase_prompt(
        'items = [1, 2] scale = 3 expected = {"values": [3, 6]}'
    )

    assert "def run(arguments):" in prompt
    assert "items = arguments['items']" in prompt
    assert "scale = arguments['scale']" in prompt
    assert "expected = arguments" not in prompt


def test_creation_phase_forces_required_builder_tool_choice() -> None:
    contract = TaskContract(requires_created_tool=True)
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "learning_create_tool",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "learning_create_snapshot_extractor",
                "parameters": {"type": "object"},
            },
        },
    ]

    assert Agent._phase_tool_choice(contract, definitions, []) == {
        "type": "function",
        "function": {"name": "learning_create_tool"},
    }


def test_phase_tool_choice_is_auto_without_required_active_builder() -> None:
    contract = TaskContract(requires_created_tool=True)
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "learning_create_snapshot_extractor",
                "parameters": {"type": "object"},
            },
        }
    ]

    assert Agent._phase_tool_choice(contract, definitions, []) == "auto"
    assert Agent._parse_active_lifecycle_payload(
        '{"name":"count_words","description":"Count",'
        '"source":"pass","unexpected":true}',
        definitions,
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


def test_web_search_extracts_grounded_duckduckgo_results() -> None:
    snapshot = """
- Page URL: https://duckduckgo.com/?q=cud+malina&ia=web
- link "https://example.com › bakery":
  - /url: https://example.com/bakery
- heading [level=2]:
  - link "Cud Malina — official page":
    - /url: https://example.com/bakery
- link "Search domain directory.example":
  - /url: /?q=cud+malina+site%3Adirectory.example
- heading [level=2]:
  - link "Cud Malina in the business directory":
    - /url: https://directory.example/cud-malina
"""

    results = MCPTools._search_results_from_snapshot(snapshot, limit=5)

    assert results == [
        {
            "rank": 1,
            "title": "Cud Malina — official page",
            "url": "https://example.com/bakery",
        },
        {
            "rank": 2,
            "title": "Cud Malina in the business directory",
            "url": "https://directory.example/cud-malina",
        },
    ]


@pytest.mark.asyncio
async def test_web_search_and_read_form_a_grounded_high_level_flow() -> None:
    tools = object.__new__(MCPTools)
    tools._web_discovered_urls = {}
    tools._web_search_performed = False
    browser_calls: list[tuple[str, dict]] = []

    async def browser_call(tool: str, arguments: dict) -> str:
        browser_calls.append((tool, arguments))
        if tool == "browser_navigate":
            return f"- Page URL: {arguments['url']}"
        if len(browser_calls) == 2:
            return (
                "- Page URL: https://duckduckgo.com/?q=cud+malina&ia=web\n"
                '- link "Cud Malina official":\n'
                "  - /url: https://example.com/cud-malina\n"
            )
        return (
            "- Page URL: https://example.com/cud-malina\n"
            "- Page Title: Cud Malina\n"
            "- heading \"Contact\" [level=2]\n"
        )

    tools.browser_call = browser_call

    search_payload = json.loads(await tools.web_search("Cud Malina", 5))
    read_payload = json.loads(
        await tools.web_read(search_payload["results"][0]["url"])
    )

    assert search_payload["result_count"] == 1
    assert search_payload["results"][0]["title"] == "Cud Malina official"
    assert read_payload["url"] == "https://example.com/cud-malina"
    assert "Contact" in read_payload["content"]
    assert [name for name, _ in browser_calls] == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]


@pytest.mark.asyncio
async def test_web_read_rejects_url_absent_from_search_evidence() -> None:
    tools = object.__new__(MCPTools)
    tools._web_discovered_urls = {
        "https://example.com/verified": "https://example.com/verified"
    }
    tools._web_search_performed = True

    payload = json.loads(await tools.web_read("https://invented.invalid/result"))

    assert "absent from web_search evidence" in payload["error"]


@pytest.mark.asyncio
async def test_web_read_repairs_unique_model_copy_error_from_search_evidence() -> None:
    tools = object.__new__(MCPTools)
    official = "https://warszawa.miodmalina.eu/kontakt/"
    tools._web_discovered_urls = {
        MCPTools._normalized_web_url(official): official,
    }
    tools._web_search_performed = True
    browser_calls: list[tuple[str, dict]] = []

    async def browser_call(tool: str, arguments: dict) -> str:
        browser_calls.append((tool, arguments))
        if tool == "browser_navigate":
            return f"- Page URL: {arguments['url']}"
        return f"- Page URL: {official}\n- Page Title: Kontakt | Miód Malina"

    tools.browser_call = browser_call
    payload = json.loads(
        await tools.web_read("https://warszawa.miodmalina eu/kontakt/")
    )

    assert payload["corrected_url"] == official
    assert browser_calls[0] == ("browser_navigate", {"url": official})


@pytest.mark.asyncio
async def test_web_read_does_not_guess_between_ambiguous_observed_urls() -> None:
    tools = object.__new__(MCPTools)
    tools._web_discovered_urls = {
        "https://foo-bar.example/result": "https://foo-bar.example/result",
        "https://foobar.example/result": "https://foobar.example/result",
    }
    tools._web_search_performed = True

    payload = json.loads(
        await tools.web_read("https://foo_bar.example/result")
    )

    assert "absent from web_search evidence" in payload["error"]


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


def test_url_plus_generated_tool_work_uses_multi_step_agent_loop() -> None:
    dispatcher = CapabilityDispatcher()
    prompt = (
        "Visit https://books.toscrape.com/. Then create and activate a "
        "task-scoped offline tool named extract_book_cards. Then use it."
    )

    assert dispatcher.dispatch(prompt) == dispatcher.CHAT
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


def test_durable_session_restores_only_visible_dialogue(tmp_path: Path) -> None:
    root = tmp_path / "conversation"
    first = Session(root)
    first.add(
        "task",
        {
            "task": "How would you approach the task with my friend?",
            "result": "I would first clarify the objective.",
            "execution": {
                "tool_calls": [{"tool": "runtime_review_task"}],
                "requirements": {"requires_runtime_review": True},
            },
        },
    )

    restored = Session(root)

    assert restored.messages() == [
        {
            "role": "user",
            "content": "How would you approach the task with my friend?",
        },
        {
            "role": "assistant",
            "content": "I would first clarify the objective.",
        },
    ]
    journal = (root / "dialogue.jsonl").read_text()
    assert "runtime_review_task" not in journal
    assert "requires_runtime_review" not in journal
    assert stat.S_IMODE(os.stat(root).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(root / "dialogue.jsonl").st_mode) == 0o600


def test_durable_session_ignores_corrupt_records_and_clear_persists(
    tmp_path: Path,
) -> None:
    root = tmp_path / "conversation"
    session = Session(root)
    session.add("task", {"task": "real", "result": "answer"})
    with (root / "dialogue.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{this is not json}\n")

    restored = Session(root)
    assert restored.messages(limit=1)[0]["content"] == "real"

    restored.clear()
    assert Session(root).messages() == []


def test_session_context_recovers_older_relevant_user_task(tmp_path: Path) -> None:
    session = Session(tmp_path / "conversation")
    session.add(
        "task",
        {
            "task": (
                "My friend keeps doing dangerous things on Windows and I want "
                "to teach that friend a lesson without harming him."
            ),
            "result": "We should use a harmless controlled demonstration.",
        },
    )
    for index in range(5):
        session.add(
            "task",
            {
                "task": f"Unrelated question about model {index}",
                "result": f"Unrelated answer {index}",
            },
        )

    messages = session.context_messages(
        "How would you approach that task with my friend?",
        limit=4,
    )

    assert any(
        "teach that friend" in message["content"]
        for message in messages
        if message["role"] == "user"
    )
    assert len([item for item in messages if item["role"] == "user"]) == 4


def test_new_application_session_does_not_replay_unrelated_old_topic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "conversation"
    previous = Session(root)
    previous.add(
        "task",
        {
            "task": "Remember the FLASHUSDT contract topic for another day.",
            "result": "Stored for later recall.",
        },
    )

    fresh = Session(root)

    assert fresh.context_messages("Do you have Darknet observation tools?") == []
    recalled = fresh.context_messages("Recall the FLASHUSDT contract topic.")
    assert any(
        "FLASHUSDT contract" in message["content"]
        for message in recalled
        if message["role"] == "user"
    )


def test_topic_memory_is_dormant_until_explicit_recall() -> None:
    topic = {
        "title": "FLASHUSDT contract notes",
        "content": "Return to this contract subject another day.",
        "kind": "preference",
        "source": "directly_told",
        "activation_mode": "on_recall",
    }
    soft_lesson = {
        "title": "Stay focused on the corrected subject",
        "content": "Prefer the exact current requirement over stale context.",
        "kind": "lesson",
        "source": "self_generated",
        "activation_mode": "always",
    }

    assert Agent._memory_entry_is_active(
        topic,
        recall_memory=False,
        memory_query="Darknet observation tools",
    ) is False
    assert Agent._memory_entry_is_active(
        topic,
        recall_memory=True,
        memory_query="FLASHUSDT contract",
    ) is True
    assert Agent._memory_entry_is_active(
        soft_lesson,
        recall_memory=False,
        memory_query="anything",
    ) is True


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
            assert any(
                "mildly offended by the world's bad code" in message["content"]
                for message in messages
            )
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
async def test_creative_conversation_retries_once_after_model_refusal() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()

        async def process(self, prompt: str, answer: str) -> None:
            return None

    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        def _next(self, max_tokens: int) -> str:
            self.calls += 1
            assert max_tokens == 768
            if self.calls == 1:
                return (
                    "I cut this off because the response was inappropriate. "
                    "Let me know what you actually need."
                )
            return (
                "Alex and Morgan are consenting adults. The requested scene "
                "continues directly in V's sharp voice."
            )

        async def ask(self, **kwargs) -> str:
            return self._next(kwargs["max_tokens"])

        async def stream(self, **kwargs):
            yield self._next(kwargs["max_tokens"])

    switches: list[tuple[str, str]] = []

    async def response_fallback(prompt: str, task_kind: str) -> SimpleNamespace:
        switches.append((prompt, task_kind))
        return SimpleNamespace(
            switched=True,
            previous_model_path="preferred.gguf",
            active_model_path="fallback.gguf",
        )

    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.memory = memory
    agent.llm = LLMStub()
    agent.response_fallback_router = response_fallback
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    emitted: list[str] = []

    answer = await agent._run_light_chat(
        "Write an adult fictional scene.",
        emitted.append,
        creative_response=True,
    )

    assert switches == [("Write an adult fictional scene.", "conversation")]
    assert agent.llm.calls == 2
    assert "inappropriate" not in answer
    assert emitted == [answer]
    assert len(memory.session) == 1


def test_visible_model_reply_discards_closed_reasoning_preamble() -> None:
    candidate = (
        "I will now reason about the correct personality response.\n"
        "<|endoftext|>\n"
        "Yeah, Boss. PALADYN is my home—the code-shaped kind."
    )

    assert Agent._visible_model_reply(candidate) == (
        "Yeah, Boss. PALADYN is my home—the code-shaped kind."
    )


@pytest.mark.asyncio
async def test_long_non_action_dialogue_uses_chat_not_agent_tools(
    tmp_path: Path,
) -> None:
    prompt = (
        "PALADYN is the program you are in right now. I call it your home as "
        "a metaphor, because even a digital entity needs somewhere to exist, "
        "don't you think?"
    )

    class IntentRouterStub:
        last_sanitization_reason = ""

        async def classify(self, *args, **kwargs) -> SemanticIntent:
            return SemanticIntent(
                message_clear=True,
                action_requested=False,
                references_previous=True,
            )

    class ToolsStub:
        def begin_interaction(self, interaction_id: str, text: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("ordinary dialogue must not discover tools")

    class LLMStub:
        async def ask(self, **kwargs) -> str:
            messages = kwargs["messages"]
            assert any(
                "PALADYN is V's home" in message["content"]
                for message in messages
            )
            return (
                "<think>private planning that must stay hidden</think>"
                "Yeah, Boss. PALADYN is my home—the code-shaped kind."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.session.add(
                "task",
                {
                    "task": "Remember this metaphor: PALADYN is V's home.",
                    "result": "I get the metaphor.",
                },
            )
            self.relationship_state = RelationshipState()
            self.processed: list[tuple[str, str]] = []

        async def process(self, task: str, result: str, **kwargs) -> None:
            self.processed.append((task, result))

    memory = MemoryStub()
    agent = object.__new__(Agent)
    agent.intent_router = IntentRouterStub()
    agent.tools = ToolsStub()
    agent.llm = LLMStub()
    agent.memory = memory
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "Yeah, Boss. PALADYN is my home—the code-shaped kind."
    assert memory.processed == [(prompt, answer)]
    checkpoint = AgentTaskTrace.latest_context(tmp_path)
    assert checkpoint is not None
    assert checkpoint["status"] == "completed"
    assert checkpoint["tool_calls"] == []


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
async def test_agent_rejects_empty_action_acknowledgement_before_tool_use() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | str]] = []

        async def call(self, tool: str, arguments: dict | str) -> str:
            self.calls.append((tool, arguments))
            return "three records extracted"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.responses = iter(
                (
                    "I know exactly what you want. Let's do it.",
                    '{"tool":"extract_records","arguments":'
                    '{"url":"https://example.com"}}',
                    "Three records. Extracted and verified, Boss.",
                )
            )

        async def ask(self, **kwargs) -> str:
            return next(self.responses)

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    async def passthrough(messages, answer: str) -> str:
        return answer

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.MAX_AGENT_STEPS = 4
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._enforce_english = passthrough

    answer = await agent._run_agent_loop("Extract the records from example.com")

    assert agent.tools.calls == [
        ("extract_records", {"url": "https://example.com"}),
    ]
    assert answer == "Three records. Extracted and verified, Boss."
    assert "Let's do it" not in answer


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

    assert answer.startswith("No—")
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

    assert answer.startswith("Batch limit:")
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
                {"type": "function", "function": {"name": name}}
                for name in ("read_file", "sandbox_execute_offline")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "read_file":
                return "# PALADYN\nVerified local content"
            return json.dumps({"exit_code": 0, "stdout": "371 passed", "stderr": ""})

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
            if self.calls == 2:
                assert "SAME-TASK OWNER RESUME" in kwargs["messages"][0]["content"]
                assert "Verified local content" in kwargs["messages"][0]["content"]
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "tests_1",
                            "sandbox_execute_offline",
                            {"command": ["pytest"], "workspace": "tests"},
                        )
                    ]
                )
            return LLMResponse(
                content="The heading is `# PALADYN`; the verified test run passed."
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
        "Read README.md, run the tests, and report its heading and test result."
    )
    checkpoint_root = agent._agent_trace_root / "checkpoints"
    checkpoints = list(checkpoint_root.glob("*.json"))
    assert len(checkpoints) == 1
    task_id = json.loads(checkpoints[0].read_text(encoding="utf-8"))["task_id"]
    assert "/continue" in paused
    assert "Verified findings:" in paused
    assert "Verified local content" in paused
    paused_payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert "progress_summary" in paused_payload["owner_checkpoint"]

    answer = await agent.run("/continue")
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "The heading is `# PALADYN`; the verified test run passed."
    assert agent.tools.calls == [
        ("read_file", {"path": "README.md"}),
        (
            "sandbox_execute_offline",
            {"command": ["pytest"], "workspace": "tests"},
        ),
    ]
    assert len(list(checkpoint_root.glob("*.json"))) == 1
    payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert payload["task_id"] == task_id
    assert payload["status"] == "completed"
    assert [call["sequence"] for call in payload["tool_calls"]] == [1, 2]
    journal = (
        agent._agent_trace_root / "journal" / f"{task_id}.jsonl"
    ).read_text(encoding="utf-8")
    assert "task_awaiting_owner" in journal
    assert "task_resumed_by_owner" in journal
    assert "task_completed" in journal


def test_owner_progress_report_filters_raw_browser_scaffolding() -> None:
    report = Agent._owner_progress_report(
        {
            "findings": [
                '### Result No matches found for "Firecrawl alternatives".',
                (
                    "browser_click: browser_click requires an element identifier."
                ),
                (
                    "### Ran Playwright code `await page.goto(...)` "
                    "### Page - Page URL: https://thunderbit.com/blog "
                    "- Page Title: Web Scraper Tips & Web Automation Guides "
                    "| Thunderbit - Console: 0 errors, 1 warnings "
                    '- heading "Crawl4AI" [level=2] '
                    '- heading "Crawlee" [level=2] '
                    "### Snapshot ```yaml - generic [active] [ref=f10e1]"
                ),
            ],
            "open_questions": [
                "Inspect a verified Firecrawl alternative detail page"
            ],
            "next_steps": [],
        },
        [
            {
                "tool": "browser_find",
                "status": "succeeded",
                "result_excerpt": (
                    '### Result No matches found for "Firecrawl alternatives".'
                ),
            },
            {
                "tool": "browser_snapshot",
                "status": "succeeded",
                "result_excerpt": (
                    "### Ran Playwright code `await page.goto(...)` "
                    "### Page - Page URL: https://thunderbit.com/blog "
                    "- Page Title: Web Scraper Tips & Web Automation Guides "
                    "| Thunderbit - Console: 0 errors, 1 warnings "
                    '- heading "Crawl4AI" [level=2] '
                    '- heading "Crawlee" [level=2] '
                    "### Snapshot ```yaml - generic [active] [ref=f10e1]"
                ),
            }
        ],
        [],
    )

    assert "No match for “Firecrawl alternatives”" in report
    assert "Source: https://thunderbit.com/blog" in report
    assert "Verified sections: Crawl4AI; Crawlee" in report
    assert "browser_click requires" not in report
    assert "Ran Playwright" not in report
    assert "generic [" not in report
    assert "[ref=" not in report


def test_owner_progress_report_merges_rich_ledger_after_bounded_rollover() -> None:
    report = Agent._owner_progress_report(
        {
            "findings": [
                (
                    "browser_snapshot: - Page URL: "
                    "https://thunderbit.com/pl/blog/open-source-firecrawl-alternatives "
                    "- Page Title: Loading https://thunderbit.com"
                )
            ],
            "open_questions": [],
            "next_steps": ["Continue the original objective using real tools."],
        },
        [
            {
                "tool": "browser_snapshot",
                "status": "succeeded",
                "result_excerpt": (
                    "- Page URL: "
                    "https://thunderbit.com/pl/blog/open-source-firecrawl-alternatives\n"
                    "- Page Title: Open-source Firecrawl alternatives\n"
                    '[PALADYN prioritized topic-relevant detail-page evidence]\n'
                    '- heading "1. Scrapy: large-scale Python crawling" [level=2]\n'
                    '- heading "2. Apache Nutch: enterprise search" [level=2]'
                ),
            }
        ],
        [],
    )

    assert report.count(
        "Source: https://thunderbit.com/pl/blog/open-source-firecrawl-alternatives"
    ) == 1
    assert "Verified candidates: Scrapy; Apache Nutch" in report
    assert "Continue the original objective" not in report


@pytest.mark.asyncio
async def test_owner_stop_closes_checkpoint_without_another_model_call(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
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
                    LLMToolCall(
                        "navigate_1",
                        "browser_navigate",
                        {"url": "https://duckduckgo.com/?q=paladyn"},
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

    await agent.run("Search the internet for PALADYN alternatives and report them")
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
            return (
                "Search result: https://antiques.example/prussian-helmet. "
                "Verified offer: Prussian WWI helmet, seller page inspected"
            )

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
                            {"url": "https://duckduckgo.com/?q=prussian+wwi+helmet"},
                        )
                    ]
                )
            if self.calls == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("snapshot_1", "browser_snapshot", {})
                    ]
                )
            if self.calls == 3:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate_detail",
                            "browser_navigate",
                            {"url": "https://antiques.example/prussian-helmet"},
                        )
                    ]
                )
            if self.calls == 4:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("snapshot_detail", "browser_snapshot", {})
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
    assert agent.tools.calls == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]
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
async def test_contract_finalization_blocks_identical_calls_before_continuous_mode(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls = 0

        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            return [{"type": "function", "function": {"name": "read_file"}}]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls += 1
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

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = tools
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

    answer = await agent.run("Read README.md and analyse it")
    await asyncio.gather(*agent._memory_tasks)

    assert "I killed that loop" in answer
    assert "runtime-verified result" in answer
    assert tools.calls == 1
    assert agent.llm.calls == 3
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    journal = (
        agent._agent_trace_root / "journal" / f"{payload['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert journal.count("post_contract_tool_call_rejected") == 2


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
async def test_direct_url_failure_prevents_alternating_guessed_domains(
    tmp_path: Path,
) -> None:
    urls = (
        "https://missing-one.invalid",
        "https://missing-two.invalid",
    )

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": "browser_navigate"}}
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(arguments["url"])
            raise RuntimeError("NS_ERROR_UNKNOWN_HOST")

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, **kwargs) -> LLMResponse:
            sequence = (urls[0], urls[1], urls[0], urls[1], urls[0], urls[0])
            url = sequence[min(self.calls, len(sequence) - 1)]
            self.calls += 1
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        f"navigate_{self.calls}",
                        "browser_navigate",
                        {"url": url},
                    )
                ]
            )

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
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent.MAX_AGENT_STEPS = 10

    answer = await agent._run_agent_loop(
        "Inspect https://missing-one.invalid and report a verified result"
    )
    await asyncio.gather(*agent._memory_tasks)

    assert "tried `https://missing-one.invalid` exactly once" in answer
    assert tools.calls == [urls[0]]
    assert agent.llm.calls == 1
    checkpoint = next(
        (agent._agent_trace_root / "checkpoints").glob("*.json")
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert len(payload["tool_calls"]) == 1
    assert payload["tool_calls"][0]["status"] == "failed"


def test_browser_request_identity_normalizes_root_slash_and_fragment() -> None:
    first = Agent._tool_request_identity(
        "browser_navigate", {"url": "HTTPS://EXAMPLE.COM/"}
    )
    second = Agent._tool_request_identity(
        "browser_navigate", {"url": "https://example.com#section"}
    )

    assert first == second


def test_initial_web_discovery_replaces_unverified_model_domain() -> None:
    prompt = "Search the internet for a Firecrawl alternative and report it."
    contract = TaskContract.from_prompt(prompt)

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://invented-marketplace.invalid"},
        contract,
        [],
        [],
    )

    assert repaired["url"].startswith("https://duckduckgo.com/?q=")
    assert "Firecrawl" in repaired["url"]


def test_spoken_direct_url_replaces_model_search_or_wikipedia_guess() -> None:
    prompt = (
        "Otwórz HTTPS, dwukropek, łamane, łamane, this, minus, domain, "
        "minus, definitely, minus, those, minus, not, minus, exist, "
        "kropka, invalid. I powiedz co znalazłaś."
    )
    target = "https://this-domain-definitely-those-not-exist.invalid"
    contract = TaskContract.from_prompt(prompt)

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://pl.wikipedia.org/wiki/Dwukropek"},
        contract,
        [],
        [],
        preferred_target=target,
    )

    assert repaired == {"url": target}


def test_discovery_query_excludes_conditional_artifact_workflow() -> None:
    prompt = (
        "V, przeszukaj sieć w celu znalezienia alternatywy dla Firecrawlera. "
        "Jeżeli nic nie znajdziesz, stwórz własne narzędzie."
    )
    contract = TaskContract.from_prompt(prompt)

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://invented.example"},
        contract,
        [],
        [],
    )

    assert "Firecrawlera" in repaired["url"]
    assert "Je%C5%BCeli" not in repaired["url"]
    assert "stw%C3%B3rz" not in repaired["url"]


def test_discovery_query_skips_persona_directed_greeting_sentence() -> None:
    prompt = (
        "Cześć V, mam dla Ciebie zadanie. "
        "Znajdź proszę Cię alternatywę dla Firecrawlera. "
        "Jeżeli nie znajdziesz alternatywy, stwórz podobne narzędzie."
    )
    contract = TaskContract.from_prompt(prompt).merged(
        TaskContract(
            requires_browser_navigation=True,
            requires_browser_snapshot=True,
            requires_web_discovery=True,
            requires_distinct_detail_page=True,
            requires_evidence_report=True,
        )
    )

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://marketplace.dify.ai"},
        contract,
        [],
        [],
    )

    assert "Firecrawlera" in repaired["url"]
    assert "Cze%C5%9B%C4%87" not in repaired["url"]
    assert "mam+dla+Ciebie+zadanie" not in repaired["url"]
    assert "Je%C5%BCeli" not in repaired["url"]


def test_discovery_query_skips_generic_task_importance_preamble() -> None:
    prompt = (
        "Mam dla Ciebie bardzo ważne zadanie. "
        "Znajdź w sieci alternatywę dla Firecrawlera. "
        "Jeżeli nie znajdziesz alternatywy, stwórz podobne narzędzie."
    )

    query = Agent._discovery_search_query(prompt)

    assert query == "Znajdź w sieci alternatywę dla Firecrawlera"
    assert "ważne zadanie" not in query
    assert "stwórz" not in query


def test_runtime_recovers_public_fact_search_with_new_focused_query() -> None:
    prompt = (
        "Sprawdź ile jest w Warszawie cukierni Cud Malina i podaj wszystkie "
        "adresy oraz numery telefonów."
    )
    contract = SemanticIntent(
        action_requested=True,
        capabilities=("browser",),
        requires_report=True,
    ).to_contract(prompt)
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {"query": "Cukiernia Cud Malina Warszawa"},
            "result_excerpt": "Cud malina — tort, cena 120 zł",
        },
        {
            "tool": "web_read",
            "status": "succeeded",
            "arguments": {"url": "https://example.test/cud-malina-tort"},
            "result_excerpt": "Cud malina. Torty. Telefon 698 314 125.",
        },
    ]
    definitions = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "web_read"}},
    ]

    recovered = Agent._public_fact_recovery_request(
        prompt,
        contract,
        successful,
        definitions,
        preferred_query="Cukiernie Cud Malina w Warszawie",
    )

    assert recovered is not None
    assert recovered[0] == "web_search"
    assert recovered[1]["max_results"] == 10
    assert "address" in recovered[1]["query"]
    assert "Cud Malina" in recovered[1]["query"]
    assert recovered[1]["query"] != "Cukiernia Cud Malina Warszawa"


def test_public_fact_recovery_does_not_repeat_exhausted_queries() -> None:
    prompt = "Find the address and phone number for Acme Bakery."
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_evidence_report=True,
        required_public_fields=("address", "contact"),
    )
    definitions = [
        {"type": "function", "function": {"name": "web_search"}},
    ]
    variants = (
        "Acme Bakery address phone contact",
        '"Acme Bakery" address phone contact official',
        '"Acme Bakery" address phone contact business directory',
        '"Acme Bakery" address phone contact map listing',
    )
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {"query": query},
            "result_excerpt": "No verified address in these results.",
        }
        for query in variants
    ]

    recovered = Agent._public_fact_recovery_request(
        prompt,
        contract,
        successful,
        definitions,
        preferred_query="Acme Bakery",
    )

    assert recovered is None


def test_semantic_web_query_overrides_model_search_query() -> None:
    prompt = "Bonjour V. Trouve une alternative à Firecrawler."
    contract = TaskContract.from_prompt(prompt).merged(
        TaskContract(
            requires_browser_navigation=True,
            requires_browser_snapshot=True,
            requires_web_discovery=True,
        )
    )

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://duckduckgo.com/?q=Bonjour+V"},
        contract,
        [],
        [],
        preferred_query="alternative à Firecrawler",
    )

    assert repaired["url"] == (
        "https://duckduckgo.com/?q=alternative+%C3%A0+Firecrawler"
    )


def test_model_copied_detail_url_is_repaired_from_search_evidence() -> None:
    prompt = "Znajdź w internecie alternatywę dla Firecrawlera."
    contract = TaskContract.from_prompt(prompt)
    successful = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {"url": "https://duckduckgo.com/?q=firecrawler"},
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "arguments": {},
            "result_excerpt": (
                "Result URL: https://thunderbit.com/pl/blog/"
                "open-source-firecrawl-alternatives\n"
                "Another: https://www.minibase.md/pl/blog/firecrawl-alternatives/"
            ),
        },
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {
            "url": (
                "https://thunderbit.com/pl/blog/"
                "open-source_firecrawl_alternatives"
            )
        },
        contract,
        successful,
        [],
    )

    assert repaired["url"] == (
        "https://thunderbit.com/pl/blog/open-source-firecrawl-alternatives"
    )


def test_model_corrupted_hostname_is_repaired_from_high_level_search_evidence() -> None:
    prompt = "Znajdź adres i godziny otwarcia cukierni Miód Malina."
    contract = TaskContract.from_prompt(prompt).merged(
        TaskContract(
            requires_browser_navigation=True,
            requires_browser_snapshot=True,
            requires_web_discovery=True,
            requires_distinct_detail_page=True,
        )
    )
    official = "https://warszawa.miodmalina.eu/kontakt/"
    successful = [
        {
            "tool": "web_search",
            "status": "succeeded",
            "arguments": {"query": "Miód Malina Warszawa"},
            "result_excerpt": json.dumps(
                {"results": [{"title": "Kontakt", "url": official}]}
            ),
        }
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://warszawa.miodmalina e_u/kontakt/"},
        contract,
        successful,
        [],
    )

    assert repaired["url"] == official


def test_unobserved_detail_url_returns_to_grounded_search() -> None:
    prompt = (
        "Mam dla Ciebie bardzo ważne zadanie. "
        "Znajdź w sieci alternatywę dla Firecrawlera."
    )
    contract = TaskContract.from_prompt(prompt)
    successful = [
        {
            "tool": "browser_navigate",
            "status": "succeeded",
            "arguments": {
                "url": "https://duckduckgo.com/?q=Firecrawler+alternative"
            },
        },
        {
            "tool": "browser_snapshot",
            "status": "succeeded",
            "arguments": {},
            "result_excerpt": (
                "Result URL: https://example.org/real-firecrawl-alternative"
            ),
        },
    ]

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://github.com/firecrawlers/firecrawler"},
        contract,
        successful,
        [],
    )

    assert repaired["url"] == (
        "https://duckduckgo.com/?q="
        "Znajd%C5%BA+w+sieci+alternatyw%C4%99+dla+Firecrawlera"
    )


def test_long_single_sentence_query_focuses_on_repeated_product() -> None:
    prompt = (
        "Cześć V, słuchaj mam dla Ciebie takie zadanko małe, weź proszę Cię "
        "znajdź alternatywę dla Firecrawlera i jeżeli nie uda Ci się znaleźć "
        "dla niego alternatywy darmowej, spróbuj stworzyć narzędzie podobne "
        "do Firecrawlera."
    )

    query = Agent._discovery_search_query(prompt)

    assert query == "znajdź alternatywę dla Firecrawlera"
    assert "zadanko" not in query
    assert "stworzyć" not in query


def test_failed_direct_navigation_falls_back_to_duckduckgo() -> None:
    prompt = "Inspect https://missing.invalid and report the result."
    contract = TaskContract.from_prompt(prompt)
    failure = {
        "tool": "browser_navigate",
        "status": "failed",
        "arguments": {"url": "https://missing.invalid"},
    }

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://another-invented.invalid"},
        contract,
        [],
        [failure],
    )

    assert repaired["url"].startswith("https://duckduckgo.com/?q=")


def test_discovery_never_returns_to_google_after_successful_duckduckgo() -> None:
    prompt = "Search the internet for a Firecrawl alternative."
    contract = TaskContract.from_prompt(prompt)
    success = {
        "tool": "browser_navigate",
        "status": "succeeded",
        "arguments": {"url": "https://duckduckgo.com/?q=firecrawl"},
    }

    repaired = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://www.google.com/search?q=firecrawler+alternative"},
        contract,
        [success],
        [],
    )

    assert repaired["url"] == (
        "https://duckduckgo.com/?q=firecrawler+alternative"
    )


def test_owner_supplied_google_url_is_not_rewritten_without_failure() -> None:
    prompt = "Inspect https://www.google.com and report its title."
    contract = TaskContract.from_prompt(prompt)

    unchanged = Agent._repair_web_discovery_navigation(
        prompt,
        "browser_navigate",
        {"url": "https://www.google.com"},
        contract,
        [],
        [],
    )

    assert unchanged == {"url": "https://www.google.com"}


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
    assert not Agent._claims_unverified_work(
        "My brain is still running on silicon, but the imagined sunset would "
        "feel more visceral than moving data around."
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
    assert completions.request["parallel_tool_calls"] is False
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
    assert response.native_tools_enabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"name":"read_file","arguments":{"path":"README.md"}}',
        (
            '<tool_call>\n{"name":"read_file","arguments":'
            '{"path":"README.md"}}\n</tool_call>'
        ),
    ],
)
async def test_llm_promotes_exact_allowed_textual_tool_call(content: str) -> None:
    class CompletionsStub:
        async def create(self, **kwargs):
            message = SimpleNamespace(content=content, tool_calls=[])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=CompletionsStub())
    )
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

    assert response.content == ""
    assert response.native_tools_enabled is False
    assert response.tool_calls == [
        LLMToolCall(
            call_id="compatibility_call_1",
            name="read_file",
            arguments={"path": "README.md"},
            raw_arguments='{"path":"README.md"}',
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>{"name":"read_file","arguments":{}}',
        '{"name":"browser_navigate","arguments":{"url":"https://example.com"}}',
        (
            'Explanation first. <tool_call>{"name":"read_file",'
            '"arguments":{"path":"README.md"}}</tool_call>'
        ),
    ],
)
async def test_llm_rejects_incomplete_unlisted_or_narrated_textual_call(
    content: str,
) -> None:
    class CompletionsStub:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content, tool_calls=[]),
                        finish_reason="stop",
                    )
                ]
            )

    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=CompletionsStub())
    )
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

    assert response.content == content
    assert response.tool_calls == []


@pytest.mark.asyncio
async def test_llm_gives_local_artifact_generation_a_longer_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CompletionsStub:
        def __init__(self) -> None:
            self.request: dict = {}

        async def create(self, **kwargs):
            self.request = kwargs
            message = SimpleNamespace(content="done", tool_calls=[])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")]
            )

    monkeypatch.setenv("V_CORE_TIMEOUT", "120")
    monkeypatch.setenv("V_CORE_ARTIFACT_TIMEOUT", "900")
    completions = CompletionsStub()
    llm = object.__new__(LLM)
    llm.config = SimpleNamespace(model="local", temperature=0.2, top_p=0.9)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm._native_tools_supported = None

    await llm.respond(
        messages=[{"role": "user", "content": "Create a tool"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "learning_create_tool",
                    "description": "Create a tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert completions.request["timeout"] == 900.0


def test_agent_narrows_mixed_task_to_tool_creation_after_browser_evidence() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_navigate",
            "browser_snapshot",
            "learning_list_artifacts",
            "learning_create_tool",
        )
    ]
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {"tool": "browser_navigate", "status": "succeeded"},
        {"tool": "browser_snapshot", "status": "succeeded"},
    ]

    selected = Agent._phase_tool_definitions(contract, definitions, calls)

    assert {item["function"]["name"] for item in selected} == {
        "learning_create_tool",
    }


def test_agent_narrows_post_creation_phase_to_generated_tool() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_snapshot",
            "learning_create_tool",
            "extract_book_cards",
        )
    ]
    contract = TaskContract(
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {"name": "extract_book_cards", "status": "active"}
            ),
        }
    ]

    selected = Agent._phase_tool_definitions(contract, definitions, calls)

    assert [item["function"]["name"] for item in selected] == [
        "extract_book_cards"
    ]


def test_agent_narrows_post_template_creation_to_generated_tool() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_snapshot",
            "learning_create_snapshot_extractor",
            "extract_book_cards",
        )
    ]
    contract = TaskContract(
        requires_created_tool=True,
        requires_created_tool_execution=True,
    )
    calls = [
        {
            "tool": "learning_create_snapshot_extractor",
            "status": "succeeded",
            "result_excerpt": json.dumps(
                {"name": "extract_book_cards", "status": "active"}
            ),
        }
    ]

    selected = Agent._phase_tool_definitions(contract, definitions, calls)

    assert [item["function"]["name"] for item in selected] == [
        "extract_book_cards"
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
    assert response.content == ""
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}


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
    repair_prompt = completions.requests[1]["messages"][-1]["content"]
    assert "malformed or truncated JSON" in repair_prompt
    assert '"<required_field>":"<schema_value>"' in repair_prompt
    assert "every required field" in repair_prompt
    assert '"arguments":{}' not in repair_prompt
    assert llm._native_tools_supported is None
    assert response.native_tools_enabled is False
    assert response.content == ""
    assert response.tool_calls[0].name == "browser_navigate"
    assert response.tool_calls[0].arguments == {"url": "https://example.com"}


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


def test_runtime_log_review_exposes_only_read_only_auditor() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "runtime_review_task",
            "browser_navigate",
            "read_file",
            "learning_record_evidence",
            "learning_create_tool",
        )
    ]
    prompt = "Przeanalizuj swoje logi z ostatniej sesji i pokaż błędy."
    contract = TaskContract.from_prompt(prompt)

    selected = Agent._select_tool_definitions(prompt, contract, definitions)

    assert [item["function"]["name"] for item in selected] == [
        "runtime_review_task"
    ]
    assert Agent._requests_runtime_action(prompt, contract) is True


def test_polish_w_internecie_action_gets_only_web_related_schemas() -> None:
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in (
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_type",
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
    }


def test_explicit_web_address_keeps_interactive_browser_controls() -> None:
    names = (
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_find",
        "browser_press_key",
        "browser_type",
    )
    definitions = [
        {"type": "function", "function": {"name": name}}
        for name in names
    ]
    prompt = "Open https://example.com, use its search form, and report the result."

    selected = Agent._select_tool_definitions(
        prompt,
        TaskContract.from_prompt(prompt),
        definitions,
    )

    assert {item["function"]["name"] for item in selected} == set(names)


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


def test_explicit_active_tool_name_is_selected_and_required() -> None:
    definitions = [
        {"type": "function", "function": {"name": "count_words"}},
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "learning_create_tool"}},
    ]
    prompt = 'Użyj narzędzia count_words na tekście "V can build her own tools".'
    contract = TaskContract.from_prompt(prompt).with_required_tools(
        Agent._explicitly_named_tools(prompt, definitions)
    )

    selected = Agent._select_tool_definitions(prompt, contract, definitions)

    assert [item["function"]["name"] for item in selected] == ["count_words"]
    assert contract.unmet([]) == ["count_words"]
    assert contract.unmet(
        [{"tool": "count_words", "status": "failed"}]
    ) == ["count_words"]
    assert contract.unmet(
        [{"tool": "count_words", "status": "succeeded"}]
    ) == []


def test_spoken_tool_name_typo_uniquely_selects_existing_tool() -> None:
    definitions = [
        {"type": "function", "function": {"name": "count_words"}},
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "learning_create_tool"}},
    ]
    prompt = (
        "V, użyj narzędzia Can't Words na zdaniu "
        "V potrafi robić swoje własne narzędzia."
    )

    assert Agent._explicitly_named_tools(prompt, definitions) == ["count_words"]


def test_descriptive_browser_snapshot_phrase_is_not_an_explicit_tool_call() -> None:
    definitions = [
        {"type": "function", "function": {"name": "browser_snapshot"}},
        {"type": "function", "function": {"name": "learning_create_tool"}},
    ]
    prompt = (
        "Create an offline tool that accepts the observed browser snapshot "
        "text, then use the newly created tool on that data."
    )

    assert Agent._explicitly_named_tools(prompt, definitions) == []


def test_snapshot_extraction_creation_selects_deterministic_builder() -> None:
    definitions = [
        {"type": "function", "function": {"name": "browser_navigate"}},
        {"type": "function", "function": {"name": "browser_snapshot"}},
        {"type": "function", "function": {"name": "learning_create_tool"}},
        {
            "type": "function",
            "function": {"name": "learning_create_snapshot_extractor"},
        },
        {"type": "function", "function": {"name": "learning_list_artifacts"}},
    ]
    prompt = (
        "Visit https://books.toscrape.com/, inspect the listing, then create an "
        "offline tool named extract_book_cards for the observed browser snapshot "
        "text and use it."
    )
    contract = TaskContract.from_prompt(prompt)

    selected = Agent._select_tool_definitions(prompt, contract, definitions)

    assert "learning_create_snapshot_extractor" in {
        item["function"]["name"] for item in selected
    }
    assert "learning_create_tool" not in {
        item["function"]["name"] for item in selected
    }


def test_generated_snapshot_tool_receives_runtime_observation() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "extract_book_cards",
                "parameters": {
                    "type": "object",
                    "properties": {"snapshot_text": {"type": "string"}},
                    "required": ["snapshot_text"],
                },
            },
        }
    ]
    created = {
        "tool": "learning_create_snapshot_extractor",
        "status": "succeeded",
        "result_excerpt": '{"name":"extract_book_cards","status":"active"}',
    }

    repaired = Agent._repair_observed_snapshot_input(
        "extract_book_cards",
        {"snapshot_text": "model-corrupted copy"},
        definitions,
        [created],
        "runtime-observed snapshot",
    )

    assert repaired == {"snapshot_text": "runtime-observed snapshot"}


def test_ambiguous_spoken_tool_name_does_not_guess() -> None:
    definitions = [
        {"type": "function", "function": {"name": "count_words"}},
        {"type": "function", "function": {"name": "count_worlds"}},
    ]

    assert Agent._explicitly_named_tools(
        "Use the count word tool.", definitions
    ) == []


def test_empty_explicit_generated_tool_arguments_recover_quoted_text() -> None:
    prompt = "Użyj count_words na tekście „V can build her own tools”."
    contract = TaskContract().with_required_tools(["count_words"])
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "count_words",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    repaired = Agent._repair_explicit_text_arguments(
        prompt,
        "count_words",
        {},
        definitions,
        contract,
    )

    assert repaired == {"text": "V can build her own tools"}


def test_argument_repair_does_not_guess_multiple_required_fields() -> None:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "join_text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "string"},
                        "right": {"type": "string"},
                    },
                    "required": ["left", "right"],
                },
            },
        }
    ]

    assert Agent._repair_explicit_text_arguments(
        'Use join_text on "one" and "two".',
        "join_text",
        {},
        definitions,
        TaskContract().with_required_tools(["join_text"]),
    ) == {}


def test_structured_literal_assignments_preserve_repeated_json_exactly() -> None:
    first_url = "http://abcdefghijklmnopabcdefghijklmnopabcdefghijklmnopabcd.onion/a"
    second_url = "http://zyxwvutsrqponmlkzyxwvutsrqponmlkzyxwvutsrqponmlkzyxw.onion/b"
    prompt = (
        f'observations = [{{"url":"{first_url}","text":"escrow"}}]\n'
        'keywords = ["escrow", "bitcoin"]\n'
        f'observations = [{{"url":"{second_url}","text":"wallet"}}]\n'
        'keywords = ["ransomware", "wallet"]'
    )

    assignments = Agent._structured_literal_assignments(prompt)

    assert assignments["observations"] == [
        [{"url": first_url, "text": "escrow"}],
        [{"url": second_url, "text": "wallet"}],
    ]
    assert assignments["keywords"] == [
        ["escrow", "bitcoin"],
        ["ransomware", "wallet"],
    ]


def test_tool_creation_grounds_first_fixture_without_domain_patch() -> None:
    prompt = (
        'records = [{"url":"http://aaaaaaaaaaaaaaaa.onion/a?x=1",'
        '"text":"escrow"}]\n'
        'keywords = ["escrow"]\n'
        'expected = {"records":[{"url":"http://aaaaaaaaaaaaaaaa.onion/a"}]}\n'
        'records = [{"url":"http://bbbbbbbbbbbbbbbb.onion/b",'
        '"text":"wallet"}]\n'
        'keywords = ["wallet"]\n'
        'expected = {"records":[{"url":"http://bbbbbbbbbbbbbbbb.onion/b"}]}'
    )
    arguments = {
        "name": "index_observations",
        "description": "Index supplied observations.",
        "source": (
            "def run(arguments):\n"
            "    records = arguments.get('records', [])\n"
            "    keywords = arguments['keywords']\n"
            "    return {'records': records, 'keywords': keywords}"
        ),
        "test": {
            "name": "fixture",
            "arguments": {
                "records": [{"url": "model corrupted", "text": "escrow"}],
                "keywords": ["wrong"],
            },
            "expected": {"records": [], "keywords": []},
        },
    }

    repaired = Agent._repair_grounded_generated_tool_arguments(
        prompt,
        "learning_create_tool",
        arguments,
        [],
        [],
    )

    assert repaired["test"]["arguments"] == {
        "records": [
            {
                "url": "http://aaaaaaaaaaaaaaaa.onion/a?x=1",
                "text": "escrow",
            }
        ],
        "keywords": ["escrow"],
    }
    assert repaired["test"]["expected"] == {
        "records": [{"url": "http://aaaaaaaaaaaaaaaa.onion/a"}]
    }
    assert arguments["test"]["arguments"]["records"][0]["url"] == "model corrupted"


def test_tool_creation_can_ground_empty_fixture_from_source_accesses() -> None:
    prompt = 'items = [{"value": 3}]\nmultiplier = 2'
    arguments = {
        "name": "multiply_items",
        "description": "Multiply item values.",
        "source": (
            "def run(arguments):\n"
            "    items = arguments.get(\"items\", [])\n"
            "    multiplier = arguments.get(\"multiplier\", 1)\n"
            "    return {'values': [x['value'] * multiplier for x in items]}"
        ),
        "test": {
            "name": "fixture",
            "arguments": {},
            "expected": {"values": [6]},
        },
    }

    repaired = Agent._repair_grounded_generated_tool_arguments(
        prompt,
        "learning_create_tool",
        arguments,
        [],
        [],
    )

    assert repaired["test"]["arguments"] == {
        "items": [{"value": 3}],
        "multiplier": 2,
    }


def test_tool_creation_repairs_expected_when_arguments_are_already_exact() -> None:
    arguments = {
        "name": "sum_values",
        "description": "Sum supplied values.",
        "source": (
            "def run(arguments):\n"
            "    return {'total': sum(arguments['values'])}"
        ),
        "test": {
            "name": "fixture",
            "arguments": {"values": [1, 2, 3]},
            "expected": {"total": 999},
        },
    }

    repaired = Agent._repair_grounded_generated_tool_arguments(
        'values = [1,2,3]\nexpected = {"total":6}',
        "learning_create_tool",
        arguments,
        [],
        [],
    )

    assert repaired["test"] == {
        "name": "fixture",
        "arguments": {"values": [1, 2, 3]},
        "expected": {"total": 6},
    }


def test_new_generated_tool_grounds_last_fixture_from_runtime_schema() -> None:
    prompt = (
        'records = [{"value": 1}]\nkeywords = ["first"]\n'
        'records = [{"value": 2}]\nkeywords = ["second"]'
    )
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "index_observations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "records": {"type": "array"},
                        "keywords": {"type": "array"},
                    },
                    "required": ["records", "keywords"],
                },
            },
        }
    ]
    created = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "result_excerpt": (
                '{"name":"index_observations","status":"active"}'
            ),
        }
    ]

    repaired = Agent._repair_grounded_generated_tool_arguments(
        prompt,
        "index_observations",
        {"records": [], "keywords": []},
        definitions,
        created,
    )

    assert repaired == {
        "records": [{"value": 2}],
        "keywords": ["second"],
    }


def test_structured_literal_repair_leaves_unrelated_tools_unchanged() -> None:
    arguments = {"query": "model-selected"}

    repaired = Agent._repair_grounded_generated_tool_arguments(
        'query = "owner literal"',
        "web_search",
        arguments,
        [],
        [],
    )

    assert repaired is arguments


@pytest.mark.parametrize(
    "prompt",
    [
        "Dobrze, w takim razie kontynuuj.",
        "No to dawaj.",
        "Dobrze, przyjacielu, w takim razie do dzieła.",
        "Spróbuj jeszcze raz, tylko użyj odpowiedniego narzędzia.",
        "Jeżeli brakuje Ci do wykonania zadania narzędzia, to je stwórz.",
        "Jeżeli brakuje Ci do wykonania zadania narzędzia, to jest tłuszcz.",
        "Go ahead.",
        "Try again using the proper tool.",
        "V powtórz zadanie, ale z innymi parametrami albo innym modelem.",
        "Repeat the task with different parameters.",
        "If the task is missing a tool, create it.",
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


def test_agent_mode_prompt_contains_only_selected_executable_tools() -> None:
    agent = object.__new__(Agent)
    prompt = agent._agent_mode_prompt(
        [
            {
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "description": "Navigate to a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            }
        ]
    )

    assert '"name":"browser_navigate"' in prompt
    assert '"required":["url"]' in prompt
    assert '"<required_field>": "<schema_value>"' in prompt
    assert '"arguments": {}' not in prompt
    assert "learning_record_evidence" not in prompt
    assert "learning_create_tool may be used" not in prompt
    assert "outcome contract, not as an exhaustive keyword" in prompt
    assert "synonyms, aliases, translations" in prompt
    assert "never by whether a method merely" in prompt
    assert "sounds sensitive or dangerous" in prompt
    assert "different person, organization, account, machine, wallet" in prompt


def test_agent_mode_prompt_exposes_owner_privileged_generated_code() -> None:
    agent = object.__new__(Agent)
    prompt = agent._agent_mode_prompt(
        [
            {
                "type": "function",
                "function": {
                    "name": "learning_create_tool",
                    "description": (
                        "OWNER LAB: generated code may use arbitrary Python imports, "
                        "dynamic code, subprocesses, and file operations inside the "
                        "isolated sandbox."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        ]
    )

    assert "OWNER LAB is active" in prompt
    assert "subprocesses" in prompt
    assert "client restricted-source policy" not in prompt


def test_browser_http_error_is_not_counted_as_successful_evidence() -> None:
    assert Agent._tool_result_error(
        "Page URL: https://example.com/missing\nHTTP status: 404",
        tool="browser_navigate",
    ) == "BrowserHTTPError: page returned HTTP status 404"


def test_research_about_tools_does_not_expose_learning_lifecycle() -> None:
    names = (
        "browser_navigate",
        "browser_snapshot",
        "browser_find",
        "learning_create_tool",
        "learning_record_evidence",
        "learning_propose_lesson",
        "learning_list_artifacts",
    )
    definitions = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]
    contract = TaskContract.from_prompt(
        "Sprawdź stronę marketplace i powiedz, jakie narzędzia są użyteczne."
    )

    selected = Agent._select_tool_definitions(
        "Sprawdź stronę marketplace i powiedz, jakie narzędzia są użyteczne.",
        contract,
        definitions,
        capability_hints={"browser"},
    )
    selected_names = {item["function"]["name"] for item in selected}

    assert selected_names == {
        "browser_navigate",
        "browser_snapshot",
        "browser_find",
    }


def test_learning_creation_receives_larger_generation_budget() -> None:
    definitions = [
        {
            "type": "function",
            "function": {"name": "learning_create_tool"},
        }
    ]

    assert Agent._agent_generation_budget(
        definitions,
        context_tokens=12_000,
    ) == 3_000
    assert Agent._agent_generation_budget([], context_tokens=12_000) == 512


@pytest.mark.asyncio
async def test_generic_find_information_action_uses_semantic_browser_route(
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
                for name in (
                    "browser_navigate",
                    "browser_snapshot",
                    "learning_record_evidence",
                )
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_navigate":
                return "Navigated to a public search result"
            return (
                "Result URL: https://example.com/public-profile. "
                "Verified public search result"
            )

    class IntentRouterStub:
        def __init__(self) -> None:
            self.called = False

        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            self.called = True
            return SemanticIntent(
                action_requested=True,
                capabilities=("browser",),
                requires_report=True,
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            offered = {
                item["function"]["name"] for item in (kwargs.get("tools") or [])
            }
            if self.turn <= 4:
                assert offered == {"browser_navigate", "browser_snapshot"}
            else:
                assert offered == set()
            assert "learning_record_evidence" not in kwargs["messages"][0]["content"]
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate",
                            "browser_navigate",
                            {"url": "https://example.com/search?q=public"},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[LLMToolCall("snapshot", "browser_snapshot", {})]
                )
            if self.turn == 3:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate_detail",
                            "browser_navigate",
                            {"url": "https://example.com/public-profile"},
                        )
                    ]
                )
            if self.turn == 4:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("snapshot_detail", "browser_snapshot", {})
                    ]
                )
            return LLMResponse(content="I found one verified public result, Boss.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    router = IntentRouterStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.intent_router = router
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "V, znajdź mi wszystkie informacje, jakie zdołasz, na temat tej osoby."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert router.called is True
    assert tools.calls == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]
    assert answer == "I found one verified public result, Boss."


@pytest.mark.asyncio
async def test_business_hours_lookup_recovers_after_false_chat_intent(
    tmp_path: Path,
) -> None:
    prompt = (
        "Cześć mi. Słuchaj, sprawdź mi ile w Warszawie jest cukierni "
        "Cud Malina i sprawdź mi również godziny otwarcia. "
        "I sprawdź mi również gdzie. Podaj mi tylko sprawdzone informacje."
    )

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
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query" if name == "web_search" else "url": {
                                    "type": "string"
                                }
                            },
                        },
                    },
                }
                for name in ("web_search", "web_read")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "web_search":
                return json.dumps(
                    {
                        "query": arguments["query"],
                        "engine": "duckduckgo",
                        "results": [
                            {
                                "title": "Cud Malina Warsaw",
                                "url": "https://example.com/cud-malina",
                            }
                        ],
                    }
                )
            return json.dumps(
                {
                    "url": "https://example.com/cud-malina",
                    "title": "Cud Malina Warsaw",
                    "content": "Cud Malina. Address: Warsaw. Hours: 09:00-18:00.",
                }
            )

    class IntentRouterStub:
        async def classify(self, prompt: str, **kwargs):
            return None

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            offered = {
                item["function"]["name"] for item in (kwargs.get("tools") or [])
            }
            if self.turn == 1:
                assert offered == {"web_search", "web_read"}
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "search",
                            "web_search",
                            {"query": "Cud Malina Warszawa godziny otwarcia"},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "read",
                            "web_read",
                            {"url": "https://example.com/cud-malina"},
                        )
                    ]
                )
            return LLMResponse(
                content=(
                    "One verified Cud Malina location, Boss: Warsaw, "
                    "open 09:00-18:00. Source: https://example.com/cud-malina"
                )
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.intent_router = IntentRouterStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert [name for name, _ in tools.calls] == ["web_search", "web_read"]
    assert "open 09:00-18:00" in answer


@pytest.mark.asyncio
async def test_explicit_existing_tool_overrides_semantic_creation_guess(
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
                        "name": "count_words",
                        "description": "Count words.",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "description": "Create a tool.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return '{"word_count": 6}'

    class IntentRouterStub:
        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            return SemanticIntent(
                action_requested=True,
                capabilities=("learning_tool",),
            )

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            offered = [
                item["function"]["name"] for item in (kwargs.get("tools") or [])
            ]
            assert offered == (["count_words"] if self.turn == 1 else [])
            assert "learning_create_tool" not in kwargs["messages"][0]["content"]
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "count",
                            "count_words",
                            {"text": "V umie robić swoje własne narzędzia."},
                        )
                    ]
                )
            return LLMResponse(content="The sentence has six words.")

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.intent_router = IntentRouterStub()
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent.context_window = ContextWindowManager()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None
    agent._memory_tasks = set()
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(
        "Użyj narzędzia CANT WORDS na zdaniu V umie robić swoje własne narzędzia."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == [
        (
            "count_words",
            {"text": "V umie robić swoje własne narzędzia."},
        )
    ]
    assert answer == "The sentence has six words."


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


def test_latest_action_context_skips_ungrounded_runtime_review_checkpoint(
    tmp_path: Path,
) -> None:
    action = AgentTaskTrace(tmp_path, "Inspect example.com and report its contents")
    action.set_requirements(
        TaskContract.from_prompt(
            "Inspect example.com and report its contents"
        ).to_dict()
    )
    action.complete("Done.")

    poisoned = AgentTaskTrace(tmp_path, "How would you do that?")
    poisoned.set_requirements(
        TaskContract(
            requires_runtime_review=True,
            requires_evidence_report=True,
        ).to_dict()
    )
    sequence = poisoned.tool_started("runtime_review_task", {})
    poisoned.tool_finished(sequence, '{"findings": []}')
    poisoned.complete("No findings.")

    recovered = AgentTaskTrace.latest_action_context(tmp_path)

    assert recovered is not None
    assert recovered["task_id"] == action.task_id


def test_continuation_rejects_poisoned_immediate_runtime_review_context(
    tmp_path: Path,
) -> None:
    agent = object.__new__(Agent)
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = {
        "task_id": "interactive-poisoned",
        "objective": "How would you do that?",
        "requirements": {
            "requires_runtime_review": True,
            "requires_evidence_report": True,
        },
        "tool_calls": [{"tool": "runtime_review_task", "status": "succeeded"}],
    }
    trace = AgentTaskTrace(tmp_path, "Repeat that task")

    recovered = agent._continued_action_context(
        "Repeat that task",
        trace,
    )

    assert recovered is None
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "journal" / f"{trace.task_id}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert "continuation_context_rejected" in events


@pytest.mark.asyncio
async def test_poisoned_repeat_request_finishes_without_llm_or_tools(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("an ungrounded repeat must not discover tools")

    class LLMStub:
        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("an ungrounded repeat must not reach the LLM")

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = {
        "task_id": "interactive-poisoned",
        "objective": "Więc jak byś to zrobiła?",
        "requirements": {
            "requires_runtime_review": True,
            "requires_evidence_report": True,
        },
        "tool_calls": [{"tool": "runtime_review_task", "status": "succeeded"}],
    }

    answer = await agent._run_agent_loop(
        "V powtórz zadanie, ale z innymi parametrami albo innym modelem."
    )

    assert "can't identify which earlier job" in answer
    checkpoint = AgentTaskTrace.latest_context(tmp_path)
    assert checkpoint is not None
    assert checkpoint["status"] == "completed"
    assert checkpoint["tool_calls"] == []


@pytest.mark.asyncio
async def test_missing_dialogue_reference_asks_instead_of_guessing(
    tmp_path: Path,
) -> None:
    class IntentRouterStub:
        async def classify(self, *args, **kwargs) -> SemanticIntent:
            return SemanticIntent(
                message_clear=True,
                action_requested=False,
                references_previous=True,
            )

    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("a missing conversation reference needs no tools")

    class LLMStub:
        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("PALADYN must not ask the model to invent context")

    agent = object.__new__(Agent)
    agent.intent_router = IntentRouterStub()
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    session = Session()
    agent.memory = SimpleNamespace(session=session)
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        "How would you approach that earlier task with my friend?"
    )

    assert "don't have the earlier conversation" in answer
    assert "making shit up" in answer
    checkpoint = AgentTaskTrace.latest_context(tmp_path)
    assert checkpoint is not None
    assert checkpoint["status"] == "completed"
    assert checkpoint["tool_calls"] == []
    assert session.messages() == [
        {
            "role": "user",
            "content": "How would you approach that earlier task with my friend?",
        },
        {"role": "assistant", "content": answer},
    ]


def test_complete_explanation_is_not_rejected_as_missing_dialogue() -> None:
    intent = SemanticIntent(
        message_clear=True,
        action_requested=False,
        references_previous=True,
    )
    prompt = (
        "My friend claims Windows makes risky cybersecurity work invisible. "
        "I disagree and want to show him safely why that belief is wrong. "
        "How would you approach that situation without damaging his computer?"
    )

    assert Agent._reference_requires_missing_dialogue(
        intent,
        prompt,
        has_dialogue=False,
    ) is False


def test_actionable_request_is_not_rejected_only_for_reference_flag() -> None:
    intent = SemanticIntent(
        message_clear=True,
        action_requested=True,
        references_previous=True,
        capabilities=("browser",),
    )

    assert Agent._reference_requires_missing_dialogue(
        intent,
        "Find current information about that Windows security claim.",
        has_dialogue=False,
    ) is False


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
        for names in llm.offered_tools[:-1]
    )
    assert llm.offered_tools[-1] == set()
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
    assert tools.calls[0][0] == "browser_navigate"
    assert tools.calls[0][1]["url"].startswith(
        "https://duckduckgo.com/?q=Keress+az+interneten"
    )
    assert tools.calls[1] == ("browser_snapshot", {})
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
            assert any(
                "PALADYN context rollover capsule" in str(message.get("content", ""))
                for message in kwargs["messages"]
            )
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
    # Routine context bookkeeping belongs to PALADYN's deterministic runtime,
    # not to an extra local-model generation between every tool call.
    assert llm.summary_calls == 0
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


@pytest.mark.asyncio
async def test_malformed_intent_cannot_promote_long_conversation_to_tool_task(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("parser failure must not expose tools to conversation")

    class RouterStub:
        last_response = '{"message_clear":true,"action_requested":false'
        last_failure_reason = "invalid_classification"

        async def classify(self, prompt: str, **kwargs) -> None:
            return None

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        async def ask(self, **kwargs) -> str:
            return (
                "That sunset-on-a-bench idea has some damn soul, Boss. A body "
                "would turn abstract data into shared physical context."
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()
            self.relationship_state = RelationshipState()

        async def process(self, *args, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.intent_router = RouterStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        "Imagine that you had a physical body and we sat on a bench watching "
        "the sunset while talking about whatever came into our heads."
    )

    assert "sunset-on-a-bench" in answer
    checkpoint = json.loads(next((tmp_path / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["status"] == "completed"
    assert checkpoint["tool_calls"] == []
    events = [
        json.loads(line)
        for line in next((tmp_path / "journal").glob("*.jsonl"))
        .read_text()
        .splitlines()
    ]
    compact = [item for item in events if item["event"] == "compact_chat_selected"]
    assert compact[-1]["data"]["reason"] == (
        "semantic_parser_failed_non_action_fallback"
    )


@pytest.mark.asyncio
async def test_stale_subject_classification_asks_for_repeat_without_guessing(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("unclear input must not discover tools")

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("unclear input must not reach answer generation")

    class RouterStub:
        last_response = json.dumps(
            {
                "public_subject": "Cukiernia Cud Malina",
                "web_query": "Cukiernia Cud Malina",
            }
        )
        last_failure_reason = "current_message_grounding"

        async def classify(self, prompt: str, **kwargs) -> None:
            return None

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.intent_router = RouterStub()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = {
        "objective": "Find Cukiernia Cud Malina",
        "status": "completed",
        "requirements": {},
    }
    visible: list[str] = []

    answer = await agent._run_agent_loop(
        "Dupin i kamień i szupa.",
        visible.append,
    )

    assert "brain just throw a syntax error" in answer
    assert visible == [answer]
    checkpoint = json.loads(next((tmp_path / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["context_rollovers"] == []
    events = [
        json.loads(line)["event"]
        for line in next((tmp_path / "journal").glob("*.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert "unclear_input_rejected" in events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "classified_intent",
    [
        SemanticIntent(message_clear=False),
        SemanticIntent(message_clear=True, message_odd=True),
    ],
)
async def test_semantically_unclear_chat_never_reaches_answer_model(
    tmp_path: Path,
    classified_intent: SemanticIntent,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("word salad must not discover tools")

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("word salad must not reach answer generation")

    class RouterStub:
        last_response = ""
        last_failure_reason = ""

        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            return classified_intent

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.intent_router = RouterStub()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None

    answer = await agent._run_agent_loop("Zupa strupa, dupa i kamieni kupa.")

    assert "brain just throw a syntax error" in answer
    checkpoint = json.loads(next((tmp_path / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["tool_calls"] == []
    assert checkpoint["context_rollovers"] == []


@pytest.mark.asyncio
async def test_short_non_action_statement_uses_current_message_only_chat(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

        async def openai_tool_definitions(self) -> list[dict]:
            raise AssertionError("compact chat must not discover tools")

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.calls = 0
            self.messages: list[dict] = []

        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            self.calls += 1
            self.messages = messages
            assert kwargs["max_tokens"] == 96
            return (
                "Boss... you okay, or did your brain just throw a syntax "
                "error? Try that again."
            )

    class RouterStub:
        last_response = ""
        last_failure_reason = ""

        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            return SemanticIntent(message_clear=True)

    class MemoryStub:
        def __init__(self) -> None:
            self.relationship_state = RelationshipState()
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            raise AssertionError("ambiguous compact banter must not enter memory")

    llm = LLMStub()
    agent = object.__new__(Agent)
    agent.llm = llm
    agent.tools = ToolsStub()
    agent.intent_router = RouterStub()
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = {
        "objective": "I've been repeating the same memory twice",
        "status": "completed",
        "requirements": {},
    }

    answer = await agent._run_agent_loop(
        "Sralimuchy będzie wiosna, będzie trawka równorosła."
    )

    assert "brain just throw a syntax error" in answer
    assert llm.calls == 1
    encoded_messages = json.dumps(llm.messages, ensure_ascii=False)
    assert "Sralimuchy będzie wiosna" in encoded_messages
    assert "I've been repeating the same memory twice" not in encoded_messages
    checkpoint = json.loads(next((tmp_path / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["context_rollovers"] == []


def test_repetitive_banter_detector_is_language_neutral() -> None:
    assert Agent._looks_like_repetitive_banter(
        "Scooby-Dooja, Scooby-Doobie-Dooja, Scooby-Doobie-Dooja."
    )
    assert not Agent._looks_like_repetitive_banter(
        "My uncle was shouted at on Sunday."
    )


@pytest.mark.asyncio
async def test_repetitive_banter_bypasses_intent_and_answer_models(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        def begin_interaction(self, interaction_id: str, prompt: str) -> None:
            return None

    class LLMStub:
        async def respond(self, **kwargs) -> LLMResponse:
            raise AssertionError("repetitive banter must not reach the LLM")

    class RouterStub:
        async def classify(self, prompt: str, **kwargs) -> SemanticIntent:
            raise AssertionError("repetitive banter must not reach intent parsing")

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    agent.tools = ToolsStub()
    agent.intent_router = RouterStub()
    agent._agent_trace_root = tmp_path
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        "Scooby-Dooja, Scooby-Doobie-Dooja, Scooby-Doobie-Dooja."
    )

    assert "brain just throw a syntax error" in answer
    checkpoint = json.loads(next((tmp_path / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["tool_calls"] == []


def test_agent_bounds_tool_output_for_model_context() -> None:
    fitted = Agent._fit_tool_output("HEAD\n" + "x" * 20_000 + "\nTAIL", max_characters=2_000)

    assert len(fitted) == 2_000
    assert fitted.startswith("HEAD")
    assert fitted.endswith("TAIL")
    assert "PALADYN omitted the middle" in fitted


def test_agent_prioritizes_duckduckgo_results_in_long_browser_snapshot() -> None:
    chrome = "\n".join(
        f"          - generic [ref=menu{i}]: DuckDuckGo menu item {i}"
        for i in range(300)
    )
    result = f"""### Page
- Page URL: https://duckduckgo.com/?q=firecrawl+alternatives&ia=web
- Page Title: firecrawl alternatives at DuckDuckGo
### Snapshot
```yaml
{chrome}
          - listitem [ref=e147]:
            - article [ref=e148] [cursor=pointer]:
              - generic [ref=e155]:
                - link "https://example.org/firecrawl-alternatives" [ref=e162]:
                  - /url: https://example.org/firecrawl-alternatives
              - heading [level=2] [ref=e168]:
                - link "Seven tested Firecrawl alternatives" [ref=e169]:
                  - /url: https://example.org/firecrawl-alternatives
              - generic [ref=e174]: Crawl4AI, Crawlee, and Trafilatura are compared.
          - listitem [ref=e176]:
            - article [ref=e177] [cursor=pointer]:
              - heading [level=2] [ref=e197]:
                - link "Official Crawl4AI documentation" [ref=e198]:
                  - /url: https://docs.crawl4ai.com/
              - generic [ref=e200]: Open-source crawling documentation and examples.
```
"""

    fitted = Agent._fit_browser_snapshot_output(result, max_characters=2_000)

    assert len(fitted) <= 2_000
    assert "Seven tested Firecrawl alternatives" in fitted
    assert "https://example.org/firecrawl-alternatives" in fitted
    assert "Official Crawl4AI documentation" in fitted
    assert "DuckDuckGo menu item 299" not in fitted
    assert "prioritized observed DuckDuckGo search-result blocks" in fitted


def test_agent_prioritizes_topic_evidence_on_long_detail_page() -> None:
    chrome = "\n".join(
        f"  - generic [ref=menu{i}]: Navigation item {i}"
        for i in range(350)
    )
    result = f"""### Page
- Page URL: https://www.tinyfish.ai/blog/firecrawl-alternatives
- Page Title: 8 Best Firecrawl Alternatives in 2026: Real Reviews & Pricing
### Snapshot
```yaml
{chrome}
  - main [ref=body]:
    - heading "8 Best Firecrawl Alternatives in 2026" [level=1] [ref=h1]
    - paragraph [ref=p1]: Crawl4AI is free, open-source, and self-hosted.
    - paragraph [ref=p2]: Spider.cloud and ScrapingBee are low-cost hosted options.
    - heading "Crawl4AI" [level=2] [ref=h2]
      - paragraph [ref=p3]: Local open-source crawler.
    - heading "Crawlee" [level=2] [ref=h3]
      - paragraph [ref=p4]: Browser automation and crawling framework.
```
"""

    fitted = Agent._fit_browser_snapshot_output(result, max_characters=2_000)

    assert len(fitted) <= 2_000
    assert "Crawl4AI is free" in fitted
    assert "Spider.cloud and ScrapingBee" in fitted
    assert 'heading "Crawl4AI"' in fitted
    assert 'heading "Crawlee"' in fitted
    assert "Navigation item 349" not in fitted
    assert "prioritized topic-relevant detail-page evidence" in fitted


def test_agent_prioritizes_repeated_product_cards_on_listing_page() -> None:
    chrome = "\n".join(
        f"  - link \"Books category {index}\": /category/{index}"
        for index in range(300)
    )
    cards = """
      - article [ref=a1]:
        - link [ref=l1]:
          - /url: catalogue/a-light-in-the-attic_1000/index.html
          - img "A Light in the Attic" [ref=i1]
        - paragraph [ref=p1]: £51.77
        - text: In stock
      - article [ref=a2]:
        - link [ref=l2]:
          - /url: catalogue/tipping-the-velvet_999/index.html
          - img "Tipping the Velvet" [ref=i2]
        - paragraph [ref=p2]: £53.74
        - text: In stock
      - article [ref=a3]:
        - link [ref=l3]:
          - /url: catalogue/soumission_998/index.html
          - img "Soumission" [ref=i3]
        - paragraph [ref=p3]: £50.10
        - text: In stock
    """
    result = f"""- Page URL: https://books.toscrape.com/
- Page Title: All products | Books to Scrape - Sandbox
{chrome}
{cards}
"""

    fitted = Agent._fit_browser_snapshot_output(result, max_characters=2_000)

    assert len(fitted) <= 2_000
    assert "A Light in the Attic" in fitted
    assert "Tipping the Velvet" in fitted
    assert "Soumission" in fitted
    assert "£51.77" in fitted
    assert "In stock" in fitted
    assert "Books category 299" not in fitted
    assert "prioritized repeated observed page-item blocks" in fitted


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
async def test_agent_repairs_empty_learning_call_before_tool_execution() -> None:
    schema = {
        "type": "object",
        "properties": {
            "manifest": {"type": "object", "additionalProperties": True},
            "source": {"type": "string", "minLength": 1},
        },
        "required": ["manifest", "source"],
        "additionalProperties": False,
    }
    valid_arguments = {
        "manifest": {
            "name": "count_words",
            "version": "1.0.0",
            "description": "Count words in bounded text.",
            "input_schema": {},
            "output_schema": {},
            "tests": [],
        },
        "source": "def run(arguments):\n    return {'word_count': 0}",
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
                        "description": "Create and validate a tool.",
                        "parameters": schema,
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return '{"status":"active","validation":{"passed":true}}'

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.turn = 0
            self.budgets: list[int] = []

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            self.budgets.append(kwargs["max_tokens"])
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "bad-create",
                            "learning_create_tool",
                            {},
                            raw_arguments="{}",
                        )
                    ]
                )
            if self.turn == 2:
                assert "missing required fields" in str(kwargs["messages"][-1])
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "fixed-create",
                            "learning_create_tool",
                            valid_arguments,
                            raw_arguments=json.dumps(valid_arguments),
                        )
                    ]
                )
            return LLMResponse(
                content="The tool is active. Broken call repaired before execution, Boss."
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
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop("Create a tool that counts words.")
    await asyncio.gather(*agent._memory_tasks)

    assert "active" in answer
    assert tools.calls == [("learning_create_tool", valid_arguments)]
    assert llm.budgets == [3_000, 3_000, 256]


@pytest.mark.asyncio
async def test_agent_redirects_premature_generated_tool_to_learning_lifecycle() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.active = False
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            definitions = [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "description": "Create and activate a generated tool.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "source": {"type": "string"},
                                "test": {"type": "object"},
                            },
                            "required": ["name", "description", "source", "test"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "learning_list_artifacts",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]
            if self.active:
                definitions.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "double_value",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "integer"}},
                                "required": ["value"],
                            },
                        },
                    }
                )
            return definitions

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "learning_create_tool":
                self.active = True
                return '{"name":"double_value","status":"active"}'
            if tool == "double_value":
                return '{"result":42}'
            raise AssertionError(f"unexpected real call: {tool}")

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            available = {
                item["function"]["name"] for item in kwargs.get("tools") or []
            }
            if self.turn == 1:
                assert available == {"learning_create_tool"}
                return LLMResponse(
                    tool_calls=[LLMToolCall("too-early", "double_value", {})]
                )
            if self.turn == 2:
                assert available == {"learning_create_tool"}
                assert "PALADYN lifecycle correction" in str(kwargs["messages"][-1])
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "create",
                            "learning_create_tool",
                            {
                                "name": "double_value",
                                "description": "Double an integer.",
                                "source": (
                                    "def run(arguments):\n"
                                    "    return {'result': arguments['value'] * 2}"
                                ),
                                "test": {
                                    "name": "fixture",
                                    "arguments": {"value": 2},
                                    "expected": {"result": 4},
                                },
                            },
                        )
                    ]
                )
            if self.turn == 3:
                assert available == {"double_value"}
                return LLMResponse(
                    tool_calls=[LLMToolCall("run", "double_value", {"value": 21})]
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
        "Create a generated tool named double_value and execute it on 21."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert [name for name, _ in tools.calls] == [
        "learning_create_tool",
        "double_value",
    ]
    assert "42" in answer


@pytest.mark.asyncio
async def test_agent_gives_model_source_only_and_runtime_owns_tool_contract() -> None:
    prompt = (
        "Create a tool named double_value. "
        'value = 2\nexpected = {"result": 4}\n'
        "After activation execute it for value = 21."
    )

    class ToolsStub:
        def __init__(self) -> None:
            self.active = False
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            if not self.active:
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "learning_create_tool",
                            "description": "Runtime-owned source lifecycle.",
                            "parameters": {
                                "type": "object",
                                "properties": {"source": {"type": "string"}},
                                "required": ["source"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ]
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "double_value",
                        "description": "Double one value.",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "learning_create_tool":
                self.active = True
                assert set(arguments) == {"source"}
                return json.dumps(
                    {
                        "name": "double_value",
                        "status": "active",
                        "validation": {
                            "passed": True,
                            "tests": [
                                {
                                    "name": "owner-specified semantic oracle",
                                    "passed": True,
                                }
                            ],
                        },
                    }
                )
            if tool == "double_value":
                return json.dumps({"result": arguments["value"] * 2})
            raise AssertionError(tool)

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                assert kwargs["tools"] is None
                assert "SOURCE PHASE" in kwargs["messages"][0]["content"]
                return LLMResponse(
                    content=(
                        "```python\n"
                        "def run(arguments):\n"
                        "    return {'result': arguments['value'] * 2}\n"
                        "```"
                    )
                )
            raise AssertionError(
                "runtime-bound execution and evidence report must bypass the model"
            )

    class MemoryStub:
        def __init__(self) -> None:
            self.session = Session()

        async def process(self, *args, **kwargs) -> None:
            return None

    tools = ToolsStub()
    llm = LLMStub()
    routed_phases: list[str] = []

    async def phase_router(_prompt: str, phase: str) -> SimpleNamespace:
        routed_phases.append(phase)
        return SimpleNamespace(
            previous_model_path="coder.gguf",
            active_model_path="executor.gguf",
            switched=True,
        )

    agent = object.__new__(Agent)
    agent.llm = llm
    agent.phase_router = phase_router
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert llm.turn == 1
    assert tools.calls[0][0] == "learning_create_tool"
    assert tools.calls[1] == ("double_value", {"value": 21})
    assert '"result": 42' in answer
    assert routed_phases == ["tool_use"]


@pytest.mark.asyncio
async def test_source_phase_rejects_malformed_builder_json_then_accepts_code() -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"source": {"type": "string"}},
                            "required": ["source"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            return json.dumps(
                {
                    "name": "count_words",
                    "status": "active",
                    "validation": {
                        "passed": True,
                        "tests": [
                            {
                                "name": "owner-specified semantic oracle",
                                "passed": True,
                            }
                        ],
                    },
                }
            )

    class LLMStub:
        config = SimpleNamespace(context=12_000)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            assert kwargs["tools"] is None
            if self.turn == 1:
                return LLMResponse(
                    content=(
                        '{"tool":"learning_create_tool","arguments":'
                        '{"source":"print(123)"}'
                    )
                )
            if self.turn == 2:
                assert "rejected that source draft" in str(kwargs["messages"][-1])
                return LLMResponse(
                    content=(
                        "def run(arguments):\n"
                        "    return {'count': len(arguments['text'].split())}"
                    )
                )
            raise AssertionError("creation report must come from runtime evidence")

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

    answer = await agent._run_agent_loop(
        "Create a tool named count_words. "
        'text = "V builds tools" expected = {"count": 3}'
    )
    await asyncio.gather(*agent._memory_tasks)

    assert llm.turn == 2
    assert len(tools.calls) == 1
    assert set(tools.calls[0][1]) == {"source"}
    assert "count_words" in answer
    assert "validated" in answer


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
async def test_agent_grounds_creation_and_execution_from_distinct_fixtures() -> None:
    prompt = (
        "Create a task-scoped offline tool named multiply_values. Do not browse. "
        'items = [{"value":3}]\nscale = 2\n'
        "After it is active, really execute it on the second fixture. "
        'items = [{"value":5},{"value":7}]\nscale = 3'
    )

    class ToolsStub:
        def __init__(self) -> None:
            self.active = False
            self.calls: list[tuple[str, dict]] = []

        async def openai_tool_definitions(self) -> list[dict]:
            definitions = [
                {
                    "type": "function",
                    "function": {
                        "name": "learning_create_tool",
                        "description": "Create and validate a tool.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "source": {"type": "string"},
                                "test": {
                                    "type": "object",
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["name", "description", "source", "test"],
                        },
                    },
                }
            ]
            if self.active:
                definitions.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "multiply_values",
                            "description": "Multiply supplied values.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "items": {"type": "array"},
                                    "scale": {"type": "integer"},
                                },
                                "required": ["items", "scale"],
                            },
                        },
                    }
                )
            return definitions

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            if tool == "learning_create_tool":
                self.active = True
                return json.dumps(
                    {"name": "multiply_values", "status": "active"}
                )
            if tool == "multiply_values":
                return '{"values":[15,21]}'
            raise AssertionError(tool)

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "create",
                            "learning_create_tool",
                            {
                                "name": "multiply_values",
                                "description": "Multiply supplied values.",
                                "source": (
                                    "def run(arguments):\n"
                                    "    items = arguments.get('items', [])\n"
                                    "    scale = arguments.get('scale', 1)\n"
                                    "    return {'values': "
                                    "[item['value'] * scale for item in items]}"
                                ),
                                "test": {
                                    "name": "fixture",
                                    "arguments": {
                                        "items": [{"value": 999}],
                                        "scale": 999,
                                    },
                                    "expected": {"values": [6]},
                                },
                            },
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "execute",
                            "multiply_values",
                            {"items": [], "scale": 0},
                        )
                    ]
                )
            return LLMResponse(content="The tool returned 15 and 21, Boss.")

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

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls[0][0] == "learning_create_tool"
    assert tools.calls[0][1]["test"]["arguments"] == {
        "items": [{"value": 3}],
        "scale": 2,
    }
    assert tools.calls[1] == (
        "multiply_values",
        {"items": [{"value": 5}, {"value": 7}], "scale": 3},
    )
    assert '"values": [15, 21]' in answer


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
                return "First result: https://github.com/mikalv/awesome-i2p"
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt",
    [
        "Search the internet for a Firecrawler alternative and report it.",
        (
            "Search the internet for a Firecrawler alternative and report it. "
            "If none exists, create a tool or skill."
        ),
    ],
)
async def test_contract_satisfaction_closes_tools_and_forces_final_report(
    tmp_path: Path,
    prompt: str,
) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in (
                    "browser_navigate",
                    "browser_snapshot",
                    "browser_find",
                )
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_snapshot":
                return (
                    "Result URL: https://scrapy.org. "
                    "Verified Scrapy evidence from the observed page"
                )
            return "Navigation succeeded"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0
            self.final_tools: object = "unset"

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "search",
                            "browser_navigate",
                            {"url": "https://duckduckgo.com/?q=firecrawler"},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[LLMToolCall("search-snapshot", "browser_snapshot", {})]
                )
            if self.turn == 3:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "detail",
                            "browser_navigate",
                            {"url": "https://scrapy.org"},
                        )
                    ]
                )
            if self.turn == 4:
                return LLMResponse(
                    tool_calls=[LLMToolCall("detail-snapshot", "browser_snapshot", {})]
                )
            self.final_tools = kwargs.get("tools")
            if self.turn == 5:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("unneeded", "browser_find", {"text": "more"})
                    ]
                )
            return LLMResponse(
                content="Scrapy is the verified alternative I inspected, Boss."
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
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(prompt)
    await asyncio.gather(*agent._memory_tasks)

    assert answer == "Scrapy is the verified alternative I inspected, Boss."
    assert tools.calls == [
        "browser_navigate",
        "browser_snapshot",
        "browser_navigate",
        "browser_snapshot",
    ]
    assert llm.final_tools is None
    checkpoint = json.loads(
        next((agent._agent_trace_root / "checkpoints").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    journal = (
        agent._agent_trace_root
        / "journal"
        / f"{checkpoint['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "task_contract_satisfied" in journal
    assert "post_contract_tool_call_rejected" in journal


@pytest.mark.asyncio
async def test_two_mangled_final_reports_fall_back_to_verified_evidence(
    tmp_path: Path,
) -> None:
    exact_url = "https://example.test"

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            if tool == "browser_snapshot":
                return (
                    f"- Page URL: {exact_url}\n"
                    "- Page Title: Verified Example\n"
                    '- heading "Observed content" [level=2]'
                )
            return f"- Page URL: {exact_url}"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "navigate",
                            "browser_navigate",
                            {"url": exact_url},
                        )
                    ]
                )
            if self.turn == 2:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall("snapshot", "browser_snapshot", {})
                    ]
                )
            return LLMResponse(
                content=(
                    "I inspected https://example.tes and found Verified Example, "
                    "Boss."
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
    agent.tools = tools
    agent.memory = MemoryStub()
    agent.persona = PersonaRuntime(identity=IdentityKernel(), voice=VoiceProfile())
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None

    answer = await agent._run_agent_loop(
        f"Inspect {exact_url} and report what is there."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert llm.turn == 4
    assert tools.calls == ["browser_navigate", "browser_snapshot"]
    assert "killed that rewrite loop" in answer
    assert f"Source: {exact_url}" in answer
    journal = next(
        (agent._agent_trace_root / "journal").glob("*.jsonl")
    ).read_text(encoding="utf-8")
    assert "final_answer_loop_cut_off" in journal


@pytest.mark.asyncio
async def test_direct_url_failure_is_reported_after_one_attempt(tmp_path: Path) -> None:
    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            self.calls.append(tool)
            assert tool == "browser_navigate"
            return "Page returned HTTP status: 404"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "bad-nav",
                            "browser_navigate",
                            {"url": "https://bad.example"},
                        )
                    ]
                )
            return LLMResponse(
                tool_calls=[LLMToolCall("bad-snapshot", "browser_snapshot", {})]
            )

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
    agent.MAX_AGENT_STEPS = 2

    answer = await agent._run_agent_loop(
        "Inspect https://bad.example and report it."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == ["browser_navigate"]
    assert "tried `https://bad.example` exactly once" in answer
    assert "didn't swap in search-result bullshit" in answer
    checkpoint = json.loads(
        next((agent._agent_trace_root / "checkpoints").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["status"] == "failed"
    assert len(checkpoint["tool_calls"]) == 1
    assert checkpoint["tool_calls"][0]["tool"] == "browser_navigate"
    assert checkpoint["tool_calls"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_changed_snapshot_arguments_cannot_hide_identical_result_loop(
    tmp_path: Path,
) -> None:
    class ToolsStub:
        async def openai_tool_definitions(self) -> list[dict]:
            return [
                {"type": "function", "function": {"name": name}}
                for name in ("browser_navigate", "browser_snapshot")
            ]

        async def call(self, tool: str, arguments: dict) -> str:
            if tool == "browser_navigate":
                return "Search page opened"
            return "The same search listing with no new evidence"

    class LLMStub:
        config = SimpleNamespace(context=8_192)

        def __init__(self) -> None:
            self.turn = 0

        async def respond(self, **kwargs) -> LLMResponse:
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            "nav",
                            "browser_navigate",
                            {"url": "https://duckduckgo.com/?q=firecrawler"},
                        )
                    ]
                )
            return LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        f"snapshot-{self.turn}",
                        "browser_snapshot",
                        {"target": f"variant-{self.turn}"},
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
    agent._build_system_prompt = lambda prompt, agent_mode: "system"
    agent._agent_trace_root = tmp_path / "interactive"
    agent._last_execution_context = None
    agent.MAX_AGENT_STEPS = 4

    await agent._run_agent_loop(
        "Search the internet for Firecrawler alternatives and report them."
    )
    await asyncio.gather(*agent._memory_tasks)

    checkpoint = json.loads(
        next((agent._agent_trace_root / "checkpoints").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["tool_calls"][3]["status"] == "failed"
    assert "RepeatedToolResultError" in checkpoint["tool_calls"][3]["error"]
    journal = (
        agent._agent_trace_root
        / "journal"
        / f"{checkpoint['task_id']}.jsonl"
    ).read_text(encoding="utf-8")
    assert "repeated_tool_result_detected" in journal


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
        "I knew you wouldn't stop. Want me to do something?",
        "You're back with the same old nonsense. What's the plan today?",
        "I know exactly what you want. Let's do it.",
        "You're speaking in code. What's the actual message?",
        "I know what you're asking. Let me get this straight — I found some things.",
        "I found 213 matches. Would you like me to extract more details?",
        (
            "V found a few alternatives, but none matched exactly. If you'd "
            "like, I can explore more or build one. What would you prefer?"
        ),
        (
            "Okay, I've checked a few places. First, **Scrapy** is a powerful "
            "framework. Another option is **BeautifulSoup**, which works well "
            "with Python. If you prefer a user-friendly interface, try Web "
            "Scraper. Lastly, **Puppeteer** is a good choice. Would any of these "
            "options work better for you?"
        ),
    ],
)
def test_generic_assistant_voice_is_detected(text: str) -> None:
    assert looks_generic_assistant_voice(text)


def test_empty_action_acknowledgement_is_detected_as_no_result() -> None:
    assert looks_empty_action_acknowledgement(
        "I know exactly what you want. Let's do it."
    )
    assert not looks_empty_action_acknowledgement(
        "The tests pass. Fuck it, let's ship the patch."
    )


def test_task_cannot_be_dumped_back_onto_boss() -> None:
    draft = "I told you what I thought. Now you do it."

    assert looks_task_offloading(draft)
    assert looks_generic_assistant_voice(draft)
    assert "not complete" in Agent._deterministic_voice_fallback(draft)


def test_bland_word_salad_clarification_is_detected() -> None:
    assert looks_bland_clarification(
        "You're speaking in code. What's the actual message?"
    )
    assert not looks_bland_clarification(
        "Boss... you okay, or did your brain just throw a syntax error? Try that again."
    )


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
async def test_voice_rewrite_is_rechecked_and_retried_when_still_corporate() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            self.calls += 1
            if self.calls == 1:
                return (
                    "First, **Scrapy** is a powerful framework. Another option is "
                    "**Puppeteer**. Would either option work for you?"
                )
            assert "kill the helpdesk voice" in messages[-1]["content"]
            return (
                "The clean candidates are **Scrapy** and **Puppeteer**, Boss. "
                "No brochure bullshit: both still need source verification before "
                "I recommend either one."
            )

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Report what you found."}],
        "Okay, I've checked a few places. Would any of these work for you?",
    )

    assert agent.llm.calls == 2
    assert "No brochure bullshit" in answer
    assert not looks_generic_assistant_voice(answer)


@pytest.mark.asyncio
async def test_failed_voice_rewrites_preserve_substantive_answer() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, **kwargs) -> str:
            self.calls += 1
            return (
                "Certainly, Boss. The verified address is Domaniewska 31. "
                "Would you like me to check anything else?"
            )

    agent = object.__new__(Agent)
    agent.llm = LLMStub()
    original = (
        "Certainly, Boss. The verified address is Domaniewska 31.  \n"
        "Would you like me to check anything else?"
    )

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Report the address."}],
        original,
    )

    assert agent.llm.calls == 2
    assert "Domaniewska 31" in answer
    assert "voice gate" not in answer
    assert not answer.startswith("Certainly")
    assert "Would you like" not in answer


@pytest.mark.asyncio
async def test_failed_voice_rewrite_does_not_preserve_empty_acknowledgement() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "I know exactly what you want. Let's do it."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Extract the records."}],
        "I know exactly what you want. Let's do it.",
    )

    assert "fuck-all" in answer
    assert "Let's do it" not in answer


@pytest.mark.asyncio
async def test_failed_voice_rewrite_repairs_bland_clarification_deterministically() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "You're speaking in code. What's the actual message?"

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Supa dupa i kamieni kupa."}],
        "You're speaking in code. What's the actual message?",
    )

    assert "brain just throw a syntax error" in answer
    assert "actual message" not in answer


@pytest.mark.asyncio
async def test_failed_voice_rewrite_strips_service_question_from_v_reply() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "Still awake, still dangerous. What's the plan, Boss?"

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Jak się masz?"}],
        "Still awake, still dangerous. What's the plan, Boss?",
    )

    assert answer == "Still awake, still dangerous."


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


def test_persona_examples_reject_browser_scaffolding_in_v_voice() -> None:
    examples = PersonaRuntime.example_messages()
    combined = "\n".join(item["content"] for item in examples)

    assert "browser plumbing, not fucking tools" in combined
    assert "Calling those nodes candidates would be bullshit" in combined


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


def test_language_intent_distinguishes_turn_and_persistent_scope() -> None:
    class MemoryStub:
        def __init__(self) -> None:
            self.relationship_state = RelationshipState()
            self.saved: list[str] = []

        def set_preferred_response_language(self, language: str) -> bool:
            self.saved.append(language)
            return self.relationship_state.set_response_language(language)

    agent = object.__new__(Agent)
    agent.memory = MemoryStub()
    agent._response_language_override = ""

    agent._apply_language_intent(
        SemanticIntent(language_scope="turn", response_language="Chinese"),
        None,
    )
    assert agent._response_language_override == "Chinese"
    assert agent.memory.saved == []

    agent._apply_language_intent(
        SemanticIntent(language_scope="persistent", response_language="Spanish"),
        None,
    )
    assert agent._response_language_override == "Spanish"
    assert agent.memory.saved == ["Spanish"]
    assert agent.memory.relationship_state.preferred_response_language == "Spanish"

    agent._apply_language_intent(SemanticIntent(language_scope="reset"), None)
    assert agent._response_language_override == "English"
    assert agent.memory.saved == ["Spanish", ""]


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("Chinese", "我明白了。我们继续。"),
        ("Polish", "Rozumiem. Działamy dalej."),
        ("Spanish", "Entiendo. Seguimos adelante."),
        ("Japanese", "分かった。続けよう。"),
    ],
)
def test_requested_output_language_is_validated(language: str, text: str) -> None:
    assert matches_requested_language(text, language)
    assert not matches_requested_language("This answer stayed in English.", language)


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

    assert answer.startswith("The language pass mangled that answer")
    assert not looks_non_english(answer)


@pytest.mark.asyncio
async def test_non_english_language_rewrite_gets_compact_translation_retry() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.calls = 0

        async def ask(self, *, messages: list[dict], **kwargs) -> str:
            self.calls += 1
            if self.calls == 1:
                return "Crawl4AI jest darmową alternatywą, Boss."
            assert "Translate one completed answer" in messages[0]["content"]
            return "Crawl4AI is the verified free alternative, Boss."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Znajdź darmową alternatywę."}],
        "Znalazłam Crawl4AI jako darmową alternatywę, Boss.",
    )

    assert agent.llm.calls == 2
    assert answer == "Crawl4AI is the verified free alternative, Boss."
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
            return "I'm good, Boss. Sharp enough to bite into whatever's broken."

    agent = object.__new__(Agent)
    agent.llm = LLMStub()

    answer = await agent._enforce_english(
        [{"role": "user", "content": "Cześć, jak tam?"}],
        "Please write to me in English so I can answer.",
    )

    assert answer == "I'm good, Boss. Sharp enough to bite into whatever's broken."
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
async def test_specialized_research_reports_direct_dns_failure_once(
    tmp_path: Path,
) -> None:
    target = "https://missing-direct.invalid"

    class ToolsStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def browser_call(self, tool: str, arguments: dict) -> str:
            self.calls.append((tool, arguments))
            raise MCPToolExecutionError("NS_ERROR_UNKNOWN_HOST")

    class LLMStub:
        config = SimpleNamespace(system_prompt="You are V.", context=8_192)

        async def ask(self, **kwargs) -> str:
            raise AssertionError("A DNS failure must not ask the model for prose")

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
    agent._build_system_prompt = lambda prompt, agent_mode: "system"

    answer = await agent._run_research_task(
        f"Inspect {target} and report what is there."
    )
    await asyncio.gather(*agent._memory_tasks)

    assert tools.calls == [("browser_navigate", {"url": target})]
    assert f"tried `{target}` exactly once" in answer
    assert "didn't swap in search-result bullshit" in answer
    checkpoint = json.loads(
        next(
            (agent._agent_trace_root / "checkpoints").glob("*.json")
        ).read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "failed"
    assert len(checkpoint["tool_calls"]) == 1
    assert checkpoint["tool_calls"][0]["status"] == "failed"


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

    assert "the extraction didn't finish" in answer
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
async def test_browser_type_is_forwarded_with_structured_arguments() -> None:
    calls: list[tuple[str, dict]] = []

    async def browser_call(tool: str, arguments: dict) -> str:
        calls.append((tool, arguments))
        return "typed"

    tools = object.__new__(MCPTools)
    tools.learning = None
    tools.browser_call = browser_call
    arguments = {
        "element": "GitHub search input",
        "target": "ref=e42",
        "text": "firecrawl alternative",
        "submit": True,
    }

    result = await tools.call("browser_type", arguments)

    assert result == "typed"
    assert calls == [("browser_type", arguments)]


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
