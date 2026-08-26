from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from v_core.autonomy import (
    AuthorizationEnvelope,
    AgentTaskTrace,
    AutonomousRunner,
    AutonomousTask,
    CheckpointStore,
    ChordDetector,
    ControlChannel,
    ControlSignal,
    ContextWindowManager,
    ExecutionMode,
    GlobalControlChannel,
    RuntimeRegistry,
    StepOutcome,
    StepResult,
    TaskBudget,
    TaskContract,
    MultilingualIntentRouter,
    SemanticIntent,
    TaskJournal,
    TaskStatus,
    parse_chord,
)
from v_core.autonomy.policy import AuthorizationDenied, AuthorizationGuard
from v_core.config import load_config


def test_task_contract_detects_polish_local_tool_and_required_use() -> None:
    contract = TaskContract.from_prompt(
        "Stwórz lokalne narzędzie count_words, a następnie użyj go."
    )

    assert contract.requires_created_tool is True
    assert contract.requires_created_tool_execution is True
    assert contract.unmet([]) == [
        "learning_create_tool",
        "generated_tool_execution",
    ]

    calls = [
        {
            "tool": "learning_create_tool",
            "status": "succeeded",
            "result_excerpt": json.dumps({"name": "count_words"}),
        },
        {
            "tool": "count_words",
            "status": "succeeded",
            "result_excerpt": json.dumps({"word_count": 6}),
        },
    ]
    assert contract.unmet(calls) == []


def test_task_contract_round_trip_preserves_dynamic_required_tools() -> None:
    contract = TaskContract(requires_evidence_report=True).with_required_tools(
        ["count_words", "count_words"]
    )

    restored = TaskContract.from_dict(contract.to_dict())
    merged = restored.merged(
        TaskContract().with_required_tools(["extract_domains"])
    )

    assert restored.required_tools == ("count_words",)
    assert merged.required_tools == ("count_words", "extract_domains")


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    task = AutonomousTask(objective="Build a report", task_id="report-1")
    task.transition(TaskStatus.RUNNING)
    envelope = AuthorizationEnvelope(
        workspace=str(tmp_path / "workspace"),
        capabilities={"read_workspace"},
    )
    store = CheckpointStore(tmp_path / "checkpoints")

    store.save(task, envelope)
    loaded = store.load("report-1")

    assert loaded is not None
    restored_task, restored_envelope = loaded
    assert restored_task.objective == "Build a report"
    assert restored_task.status == TaskStatus.RUNNING
    assert restored_envelope.capabilities == {"read_workspace"}


def test_journal_is_append_only_jsonl(tmp_path: Path) -> None:
    journal = TaskJournal(tmp_path / "journal")

    journal.append("task-1", "started", {"value": 1})
    journal.append("task-1", "completed", {"value": 2})

    records = journal.read("task-1")
    assert [record["event"] for record in records] == ["started", "completed"]
    assert records[1]["data"]["value"] == 2
    assert (tmp_path / "journal").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "journal" / "task-1.jsonl").stat().st_mode & 0o777 == 0o600


def test_interactive_agent_trace_records_runtime_evidence(tmp_path: Path) -> None:
    trace = AgentTaskTrace(tmp_path / "interactive", "Inspect a page")
    sequence = trace.tool_started(
        "browser_navigate",
        {"url": "https://example.com"},
    )
    trace.tool_finished(sequence, "Example Domain")
    trace.complete("The page title is Example Domain.")

    checkpoint = (
        tmp_path
        / "interactive"
        / "checkpoints"
        / f"{trace.task_id}.json"
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    records = trace._journal.read(trace.task_id)

    assert payload["status"] == "completed"
    assert payload["tool_calls"][0]["status"] == "succeeded"
    assert payload["tool_calls"][0]["result_sha256"]
    assert [record["event"] for record in records] == [
        "task_started",
        "tool_started",
        "tool_completed",
        "task_completed",
    ]
    assert trace.evidence()["successful_tool_count"] == 1
    assert checkpoint.stat().st_mode & 0o777 == 0o600


def test_interactive_trace_recovers_latest_runtime_context(tmp_path: Path) -> None:
    root = tmp_path / "interactive"
    trace = AgentTaskTrace(root, "Inspect the first result")
    trace.set_requirements({"requires_distinct_detail_page": True})
    sequence = trace.tool_started("browser_navigate", {"url": "https://bad.invalid"})
    trace.tool_finished(
        sequence,
        "DNS resolution failed",
        error="MCPToolExecutionError: DNS resolution failed",
    )
    trace.block("navigation failed")

    context = AgentTaskTrace.latest_context(root)

    assert context is not None
    assert context["objective"] == "Inspect the first result"
    assert context["status"] == "blocked"
    assert context["requirements"]["requires_distinct_detail_page"] is True
    assert context["tool_calls"][0]["error"].endswith("DNS resolution failed")


def test_task_contract_detects_generic_online_work_without_explicit_url() -> None:
    contract = TaskContract.from_prompt(
        "Znajdź w sieci narzędzia do monitorowania darknetu i podaj wyniki."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_evidence_report is True
    assert contract.unmet([]) == ["browser_navigate", "browser_snapshot"]


def test_task_contract_detects_polish_infinitive_przeszukac_internet() -> None:
    contract = TaskContract.from_prompt(
        "Masz przeszukać internet i znaleźć narzędzia oraz umiejętności "
        "potrzebne do monitorowania darknetowych forów."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_evidence_report is True


def test_task_contract_detects_polish_locative_w_internecie() -> None:
    contract = TaskContract.from_prompt(
        "Znajdź mi w internecie informacje o tej osobie i podaj wyniki."
    )

    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_evidence_report is True


@pytest.mark.asyncio
async def test_rollover_caps_complete_request_to_local_context_budget() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "completed": ["Checked source"] * 20,
                    "findings": ["Concrete verified finding " + "x" * 500] * 20,
                    "open_questions": ["Which result is relevant?"] * 20,
                    "next_steps": ["Inspect the next source"] * 20,
                }
            )

    manager = ContextWindowManager()
    contract = TaskContract(
        requires_browser_navigation=True,
        requires_browser_snapshot=True,
    )
    messages = [
        {"role": "system", "content": "S" * 12_000},
        {"role": "user", "content": "history " * 4_000},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"browser_tool_{index}",
                "description": "D" * 500,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for index in range(6)
    ]
    evidence = [
        {
            "sequence": index,
            "tool": "browser_snapshot",
            "arguments": {},
            "status": "succeeded",
            "result_excerpt": f"result-{index} " + "e" * 2_000,
            "error": "",
        }
        for index in range(10)
    ]

    rollover = await manager.rollover(
        llm=LLMStub(),
        system_prompt="S" * 12_000,
        objective="Inspect sources",
        contract=contract,
        messages=messages,
        tools=tools,
        evidence=evidence,
        previous_summary=None,
        context_tokens=12_000,
        step=10,
    )

    assert rollover.estimated_tokens_after <= int(12_000 * 0.72)
    assert rollover.estimated_tokens_after < rollover.estimated_tokens_before
    assert rollover.evidence
    assert all(
        len(item) <= 240
        for values in rollover.summary.values()
        for item in values
    )


@pytest.mark.asyncio
async def test_rollover_cannot_relabel_failed_runtime_call_as_success() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return json.dumps(
                {
                    "completed": ["Created and executed count_words"],
                    "findings": ["Sandbox problem was fixed and execution succeeded"],
                    "open_questions": ["What should run next?"],
                    "next_steps": ["Use a different strategy"],
                }
            )

    evidence = [
        {
            "sequence": 1,
            "tool": "learning_create_tool",
            "arguments": {},
            "status": "failed",
            "result_excerpt": (
                "ArtifactValidationError: bwrap loopback RTM_NEWADDR denied"
            ),
            "error": "ArtifactValidationError",
        }
    ]
    rollover = await ContextWindowManager().rollover(
        llm=LLMStub(),
        system_prompt="system",
        objective="Create count_words",
        contract=TaskContract(requires_created_tool=True),
        messages=[{"role": "user", "content": "Create count_words"}],
        tools=[],
        evidence=evidence,
        previous_summary=None,
        context_tokens=8_192,
        step=1,
    )

    assert rollover.summary["completed"] == ["learning_create_tool: failed"]
    assert "RTM_NEWADDR" in rollover.summary["findings"][0]
    assert not any(
        "succeeded" in item for item in rollover.summary["findings"]
    )


def test_semantic_intent_accepts_only_closed_capability_vocabulary() -> None:
    intent = SemanticIntent.parse(
        json.dumps(
            {
                "action_requested": True,
                "continue_previous": False,
                "capabilities": ["browser", "invented_shell", "browser"],
                "requires_report": True,
                "distinct_detail_page": True,
            }
        )
    )

    assert intent is not None
    assert intent.capabilities == ("browser",)
    contract = intent.to_contract()
    assert contract.requires_browser_navigation is True
    assert contract.requires_browser_snapshot is True
    assert contract.requires_distinct_detail_page is True
    assert contract.requires_evidence_report is True


@pytest.mark.asyncio
async def test_multilingual_intent_router_classifies_hungarian_action() -> None:
    class LLMStub:
        def __init__(self) -> None:
            self.kwargs: dict = {}

        async def ask(self, **kwargs) -> str:
            self.kwargs = kwargs
            return json.dumps(
                {
                    "action_requested": True,
                    "continue_previous": False,
                    "capabilities": ["browser"],
                    "requires_report": True,
                    "distinct_detail_page": False,
                }
            )

    llm = LLMStub()
    router = MultilingualIntentRouter(llm)

    intent = await router.classify(
        "Keress az interneten es keszits jelentest az eredmenyekrol."
    )

    assert intent is not None
    assert intent.action_requested is True
    assert intent.capabilities == ("browser",)
    assert intent.requires_report is True
    assert llm.kwargs["temperature"] == 0.0
    assert llm.kwargs["max_tokens"] == 128


@pytest.mark.asyncio
async def test_multilingual_intent_router_fails_closed_on_invalid_output() -> None:
    class LLMStub:
        async def ask(self, **kwargs) -> str:
            return "I think this probably needs a browser."

    intent = await MultilingualIntentRouter(LLMStub()).classify(
        "Keress az interneten."
    )

    assert intent is None


def test_task_contract_requires_real_detail_page_after_search_listing() -> None:
    contract = TaskContract.from_prompt(
        "Open https://github.com/search?q=i2p and inspect the first result."
    )
    listing_only = [
        {
            "tool": "browser_navigate",
            "arguments": {"url": "https://github.com/search?q=i2p"},
            "status": "succeeded",
        },
        {"tool": "browser_snapshot", "arguments": {}, "status": "succeeded"},
    ]

    assert contract.requires_distinct_detail_page is True
    assert contract.unmet(listing_only) == ["browser_navigate:distinct_detail_page"]


def test_task_contract_requires_real_file_and_command_actions() -> None:
    read_contract = TaskContract.from_prompt(
        "Read README.md and report only its first heading."
    )
    write_contract = TaskContract.from_prompt("Write the report to result.md")
    command_contract = TaskContract.from_prompt("Run the pytest tests and report results")

    assert read_contract.requires_file_read is True
    assert read_contract.requires_first_heading is True
    assert write_contract.unmet([]) == ["filesystem_mutation"]
    assert command_contract.unmet([]) == ["command_execution"]


def test_authorization_guard_prevents_workspace_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "tasks" / "one"
    envelope = AuthorizationEnvelope(workspace=str(workspace))
    guard = AuthorizationGuard(tmp_path, envelope)

    assert guard.resolve_task_path("report.md", write=True) == (
        workspace / "report.md"
    ).resolve()

    with pytest.raises(AuthorizationDenied):
        guard.resolve_task_path("../../outside.txt", write=True)


def test_read_only_mode_rejects_writes(tmp_path: Path) -> None:
    envelope = AuthorizationEnvelope(
        mode=ExecutionMode.READ_ONLY,
        workspace=str(tmp_path / "workspace"),
    )
    guard = AuthorizationGuard(tmp_path, envelope)

    with pytest.raises(AuthorizationDenied):
        guard.resolve_task_path("result.txt", write=True)


def test_autonomy_root_can_be_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PALADYN_AUTONOMY_ROOT", "state/tasks")
    monkeypatch.setenv("PALADYN_MODEL_RUNTIME_ROOT", "state/models")
    monkeypatch.setenv("PALADYN_VOICE_ROOT", "state/voice")
    monkeypatch.setenv("PALADYN_MODEL_LOADER", "required")
    monkeypatch.setenv("V_CORE_MCP_FILESYSTEM", "workspace")

    config = load_config()

    assert config.autonomy_root == (tmp_path / "state/tasks").resolve()
    assert config.model_runtime_root == (tmp_path / "state/models").resolve()
    assert config.voice_root == (tmp_path / "state/voice").resolve()
    assert config.model_loader_mode == "required"
    assert config.workspace == (tmp_path / "workspace").resolve()


@pytest.mark.asyncio
async def test_runner_completes_multi_step_task(tmp_path: Path) -> None:
    runner = AutonomousRunner(tmp_path / "autonomy", poll_interval=0.01)
    task = AutonomousTask(objective="Three steps", task_id="multi-step")
    envelope = AuthorizationEnvelope(
        workspace=str(tmp_path / "workspace"),
        budget=TaskBudget(max_actions=5),
    )

    async def driver(task: AutonomousTask, envelope: AuthorizationEnvelope) -> StepResult:
        if task.action_count < 3:
            return StepResult(
                StepOutcome.CONTINUE,
                summary=f"step {task.action_count}",
            )
        return StepResult(StepOutcome.COMPLETE, result="done")

    result = await runner.run(task, envelope, driver)

    assert result.status == TaskStatus.COMPLETED
    assert result.action_count == 3
    assert result.result == "done"
    assert runner.checkpoints.load("multi-step") is not None


@pytest.mark.asyncio
async def test_runner_stops_at_action_budget(tmp_path: Path) -> None:
    runner = AutonomousRunner(tmp_path / "autonomy", poll_interval=0.01)
    task = AutonomousTask(objective="Never ends", task_id="budgeted")
    envelope = AuthorizationEnvelope(
        workspace=str(tmp_path / "workspace"),
        budget=TaskBudget(max_actions=2),
    )

    async def driver(task: AutonomousTask, envelope: AuthorizationEnvelope) -> StepResult:
        return StepResult(StepOutcome.CONTINUE)

    result = await runner.run(task, envelope, driver)

    assert result.status == TaskStatus.BLOCKED
    assert result.action_count == 2
    assert result.blocked_reason == "maximum action count reached"


@pytest.mark.asyncio
async def test_stop_cancels_active_step(tmp_path: Path) -> None:
    root = tmp_path / "autonomy"
    runner = AutonomousRunner(root, poll_interval=0.01)
    task = AutonomousTask(objective="Long operation", task_id="stop-active")
    envelope = AuthorizationEnvelope(workspace=str(tmp_path / "workspace"))
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def driver(task: AutonomousTask, envelope: AuthorizationEnvelope) -> StepResult:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return StepResult(StepOutcome.COMPLETE)

    running = asyncio.create_task(runner.run(task, envelope, driver))
    await asyncio.wait_for(started.wait(), timeout=1)
    ControlChannel(root / "control", task.task_id).request(ControlSignal.STOP)
    result = await asyncio.wait_for(running, timeout=1)

    assert result.status == TaskStatus.STOPPED
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_panic_cancels_active_step(tmp_path: Path) -> None:
    root = tmp_path / "autonomy"
    runner = AutonomousRunner(root, poll_interval=0.01)
    task = AutonomousTask(objective="Long operation", task_id="panic-active")
    envelope = AuthorizationEnvelope(workspace=str(tmp_path / "workspace"))
    started = asyncio.Event()

    async def driver(task: AutonomousTask, envelope: AuthorizationEnvelope) -> StepResult:
        started.set()
        await asyncio.sleep(30)
        return StepResult(StepOutcome.COMPLETE)

    running = asyncio.create_task(runner.run(task, envelope, driver))
    await asyncio.wait_for(started.wait(), timeout=1)
    ControlChannel(root / "control", task.task_id).request(ControlSignal.PANIC)
    result = await asyncio.wait_for(running, timeout=1)

    assert result.status == TaskStatus.PANICKED
    assert "panic requested" in result.blocked_reason


@pytest.mark.asyncio
async def test_pause_and_resume_before_step(tmp_path: Path) -> None:
    root = tmp_path / "autonomy"
    runner = AutonomousRunner(root, poll_interval=0.01)
    task = AutonomousTask(objective="Pause first", task_id="pause-first")
    envelope = AuthorizationEnvelope(workspace=str(tmp_path / "workspace"))
    channel = ControlChannel(root / "control", task.task_id, poll_interval=0.01)
    channel.request(ControlSignal.PAUSE)

    async def driver(task: AutonomousTask, envelope: AuthorizationEnvelope) -> StepResult:
        return StepResult(StepOutcome.COMPLETE, result="resumed")

    running = asyncio.create_task(runner.run(task, envelope, driver))
    for _ in range(50):
        await asyncio.sleep(0.01)
        loaded = runner.checkpoints.load(task.task_id)
        if loaded and loaded[0].status == TaskStatus.PAUSED:
            break
    else:
        pytest.fail("task did not enter paused state")

    channel.request(ControlSignal.RESUME)
    result = await asyncio.wait_for(running, timeout=1)

    assert result.status == TaskStatus.COMPLETED
    assert result.result == "resumed"


def test_emergency_chord_requires_simultaneous_key_state() -> None:
    detector = ChordDetector(parse_chord("Q+P+0"))

    assert not detector.feed("Q", True)
    assert not detector.feed("P", True)
    assert detector.feed("0", True)
    assert not detector.feed("0", True)
    assert not detector.feed("P", False)
    assert detector.feed("P", True)


def test_runtime_registry_uses_pid_and_process_start_identity(tmp_path: Path) -> None:
    registry = RuntimeRegistry(tmp_path / "runtime")
    path = registry.register("test-runtime")

    assert path.exists()
    assert registry.active()[0]["name"] == "test-runtime"
    assert registry.terminate_all() == []  # never terminate the watcher itself

    registry.unregister()
    assert registry.active() == []


@pytest.mark.asyncio
async def test_runtime_registry_terminates_registered_process(tmp_path: Path) -> None:
    process = await asyncio.create_subprocess_exec("/usr/bin/sleep", "30")
    registry = RuntimeRegistry(tmp_path / "runtime")
    registry.register("disposable-test-process", pid=process.pid)

    terminated = registry.terminate_all()
    await asyncio.wait_for(process.wait(), timeout=1)

    assert terminated == [process.pid]
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_global_panic_cancels_every_active_runner(tmp_path: Path) -> None:
    root = tmp_path / "autonomy"
    runner = AutonomousRunner(root, poll_interval=0.01)
    tasks = [
        AutonomousTask(objective="long", task_id="global-one"),
        AutonomousTask(objective="long", task_id="global-two"),
    ]
    started = [asyncio.Event(), asyncio.Event()]

    def driver(index: int):
        async def run(
            task: AutonomousTask,
            envelope: AuthorizationEnvelope,
        ) -> StepResult:
            started[index].set()
            await asyncio.sleep(30)
            return StepResult(StepOutcome.COMPLETE)

        return run

    running = [
        asyncio.create_task(
            runner.run(
                task,
                AuthorizationEnvelope(workspace=str(tmp_path / task.task_id)),
                driver(index),
            )
        )
        for index, task in enumerate(tasks)
    ]
    await asyncio.gather(*(event.wait() for event in started))
    GlobalControlChannel(root / "control").request_panic()
    results = await asyncio.wait_for(asyncio.gather(*running), timeout=1)

    assert {result.status for result in results} == {TaskStatus.PANICKED}
