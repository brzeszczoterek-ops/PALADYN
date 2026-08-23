from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .control import ControlChannel, PanicRequested, StopRequested
from .journal import CheckpointStore, TaskJournal
from .models import (
    AuthorizationEnvelope,
    AutonomousTask,
    ControlSignal,
    StepOutcome,
    StepResult,
    TaskStatus,
)


StepDriver = Callable[
    [AutonomousTask, AuthorizationEnvelope],
    Awaitable[StepResult],
]
LearningSink = Callable[[str, str, dict], object]


class AutonomousRunner:
    def __init__(
        self,
        root: Path,
        *,
        poll_interval: float = 0.25,
        learning_sink: LearningSink | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = TaskJournal(self.root / "journal")
        self.checkpoints = CheckpointStore(self.root / "checkpoints")
        self.control_root = self.root / "control"
        self.poll_interval = poll_interval
        self.learning_sink = learning_sink

    async def run(
        self,
        task: AutonomousTask,
        envelope: AuthorizationEnvelope,
        driver: StepDriver,
    ) -> AutonomousTask:
        control = ControlChannel(
            self.control_root,
            task.task_id,
            poll_interval=self.poll_interval,
        )
        control.clear(ControlSignal.RESUME)

        if task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.PANICKED,
        }:
            return task

        task.transition(TaskStatus.RUNNING)
        self.journal.append(
            task.task_id,
            "task_started",
            {
                "objective": task.objective,
                "mode": envelope.mode.value,
                "budget": asdict(envelope.budget),
            },
        )
        self._save(task, envelope)

        try:
            while True:
                await self._handle_control(task, envelope, control)
                budget_error = self._budget_error(task, envelope)
                if budget_error:
                    task.blocked_reason = budget_error
                    task.transition(TaskStatus.BLOCKED)
                    self.journal.append(
                        task.task_id,
                        "budget_exhausted",
                        {"reason": budget_error},
                    )
                    self._save(task, envelope)
                    return task

                task.action_count += 1
                task.touch()
                self.journal.append(
                    task.task_id,
                    "step_started",
                    {"action_number": task.action_count},
                )
                self._save(task, envelope)

                try:
                    result = await self._run_step_with_control(
                        task,
                        envelope,
                        driver,
                        control,
                    )
                except (StopRequested, PanicRequested):
                    raise
                except Exception as exc:
                    task.failure_count += 1
                    task.touch()
                    self.journal.append(
                        task.task_id,
                        "step_failed",
                        {
                            "action_number": task.action_count,
                            "failure_number": task.failure_count,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    self._capture_learning(
                        task,
                        "step_failed",
                        {
                            "action_number": task.action_count,
                            "failure_number": task.failure_count,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    self._save(task, envelope)

                    if task.failure_count > envelope.budget.max_failures:
                        task.blocked_reason = "maximum failure count exceeded"
                        task.transition(TaskStatus.FAILED)
                        self.journal.append(
                            task.task_id,
                            "task_failed",
                            {"reason": task.blocked_reason},
                        )
                        self._save(task, envelope)
                        return task

                    await asyncio.sleep(min(0.25 * task.failure_count, 5.0))
                    continue

                self.journal.append(
                    task.task_id,
                    "step_completed",
                    {
                        "action_number": task.action_count,
                        "outcome": result.outcome.value,
                        "summary": result.summary,
                        "metadata": result.metadata,
                    },
                )

                if result.next_instruction:
                    task.metadata["next_instruction"] = result.next_instruction

                if result.outcome == StepOutcome.COMPLETE:
                    task.result = result.result or result.summary
                    task.transition(TaskStatus.COMPLETED)
                    self.journal.append(
                        task.task_id,
                        "task_completed",
                        {"result": task.result},
                    )
                    self._capture_learning(
                        task,
                        "task_completed",
                        {
                            "result": task.result,
                            "verified": bool(result.metadata.get("verified", False)),
                            "expected": result.metadata.get("expected", ""),
                            "actual": result.metadata.get("actual", task.result),
                        },
                    )
                    self._save(task, envelope)
                    return task

                if result.outcome == StepOutcome.BLOCKED:
                    task.blocked_reason = result.summary or result.result
                    task.transition(TaskStatus.BLOCKED)
                    self.journal.append(
                        task.task_id,
                        "task_blocked",
                        {"reason": task.blocked_reason},
                    )
                    self._capture_learning(
                        task,
                        "task_blocked",
                        {"reason": task.blocked_reason},
                    )
                    self._save(task, envelope)
                    return task

                task.touch()
                self._save(task, envelope)

        except StopRequested as exc:
            task.blocked_reason = str(exc)
            task.transition(TaskStatus.STOPPED)
            self.journal.append(task.task_id, "task_stopped", {"reason": str(exc)})
            self._save(task, envelope)
            return task
        except PanicRequested as exc:
            task.blocked_reason = str(exc)
            task.transition(TaskStatus.PANICKED)
            self.journal.append(task.task_id, "task_panicked", {"reason": str(exc)})
            self._save(task, envelope)
            return task
        except asyncio.CancelledError:
            task.blocked_reason = "runner task cancelled"
            task.transition(TaskStatus.STOPPED)
            self.journal.append(
                task.task_id,
                "task_cancelled",
                {"reason": task.blocked_reason},
            )
            self._save(task, envelope)
            raise

    async def resume(
        self,
        task_id: str,
        driver: StepDriver,
    ) -> AutonomousTask:
        loaded = self.checkpoints.load(task_id)
        if loaded is None:
            raise FileNotFoundError(f"no checkpoint for task {task_id}")
        task, envelope = loaded
        return await self.run(task, envelope, driver)

    async def _handle_control(
        self,
        task: AutonomousTask,
        envelope: AuthorizationEnvelope,
        control: ControlChannel,
    ) -> None:
        pending = control.pending()
        if ControlSignal.PAUSE in pending:
            task.transition(TaskStatus.PAUSED)
            self.journal.append(task.task_id, "task_paused")
            self._save(task, envelope)
            resumed = await control.checkpoint()
            if resumed:
                task.transition(TaskStatus.RUNNING)
                self.journal.append(task.task_id, "task_resumed")
                self._save(task, envelope)
                return

        await control.checkpoint()

    async def _run_step_with_control(
        self,
        task: AutonomousTask,
        envelope: AuthorizationEnvelope,
        driver: StepDriver,
        control: ControlChannel,
    ) -> StepResult:
        step_task = asyncio.create_task(driver(task, envelope))

        try:
            while not step_task.done():
                pending = control.pending()
                if ControlSignal.PANIC in pending:
                    step_task.cancel()
                    await asyncio.gather(step_task, return_exceptions=True)
                    raise PanicRequested("panic requested during active step")
                if ControlSignal.STOP in pending:
                    step_task.cancel()
                    await asyncio.gather(step_task, return_exceptions=True)
                    raise StopRequested("stop requested during active step")

                await asyncio.wait(
                    {step_task},
                    timeout=self.poll_interval,
                )

            return await step_task
        except asyncio.CancelledError:
            if not step_task.done():
                step_task.cancel()
                await asyncio.gather(step_task, return_exceptions=True)
            raise

    def _save(
        self,
        task: AutonomousTask,
        envelope: AuthorizationEnvelope,
    ) -> None:
        task.checkpoint_sequence += 1
        task.touch()
        self.checkpoints.save(task, envelope)

    def _capture_learning(
        self,
        task: AutonomousTask,
        event: str,
        data: dict,
    ) -> None:
        if self.learning_sink is None:
            return
        try:
            self.learning_sink(task.task_id, event, data)
        except Exception as exc:
            self.journal.append(
                task.task_id,
                "learning_capture_failed",
                {
                    "event": event,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

    @staticmethod
    def _budget_error(
        task: AutonomousTask,
        envelope: AuthorizationEnvelope,
    ) -> str:
        budget = envelope.budget
        if task.action_count >= budget.max_actions:
            return "maximum action count reached"
        if task.failure_count > budget.max_failures:
            return "maximum failure count exceeded"

        if task.started_at:
            started = datetime.fromisoformat(task.started_at)
            elapsed = (datetime.now(UTC) - started).total_seconds()
            if elapsed >= budget.max_runtime_seconds:
                return "maximum runtime reached"

        workspace = Path(envelope.workspace)
        if workspace.exists():
            used_bytes = sum(
                path.stat().st_size
                for path in workspace.rglob("*")
                if path.is_file()
            )
            if used_bytes > budget.max_disk_usage_mb * 1024 * 1024:
                return "maximum task workspace size exceeded"

        return ""
