from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ExecutionMode(str, Enum):
    GUIDED = "guided"
    OWNER = "owner"
    SANDBOX = "sandbox"
    READ_ONLY = "read_only"
    NO_TOOLS = "no_tools"
    FULL_AUTONOMOUS = "full_autonomous"


class TaskStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    PANICKED = "panicked"


class ControlSignal(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    PANIC = "panic"


class StepOutcome(str, Enum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    BLOCKED = "blocked"


@dataclass(slots=True)
class TaskBudget:
    max_runtime_seconds: int = 36_000
    max_actions: int = 500
    max_failures: int = 20
    max_disk_usage_mb: int = 20_480
    max_external_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if self.max_failures < 0:
            raise ValueError("max_failures cannot be negative")
        if self.max_disk_usage_mb <= 0:
            raise ValueError("max_disk_usage_mb must be positive")
        if self.max_external_cost < 0:
            raise ValueError("max_external_cost cannot be negative")


@dataclass(slots=True)
class AuthorizationEnvelope:
    mode: ExecutionMode = ExecutionMode.FULL_AUTONOMOUS
    workspace: str = "agent_workspace/autonomous"
    capabilities: set[str] = field(
        default_factory=lambda: {
            "read_workspace",
            "write_task_workspace",
            "browse_public_sources",
            "create_task_tools",
            "record_learning_evidence",
            "activate_task_artifacts",
            "run_task_tests",
            "run_sandboxed_code",
            "restart_local_models",
        }
    )
    owner_approved_capabilities: set[str] = field(default_factory=set)
    protected_paths: tuple[str, ...] = (
        "src/v_core/persona",
        "src/v_core/learning/policy.py",
        "src/v_core/learning/tool_host.py",
        "src/v_core/model_loader",
        "src/v_core/autonomy/control.py",
        "src/v_core/autonomy/models.py",
        "learning",
        "model_runtime",
        "memory",
        ".env",
    )
    allow_external_models: bool = False
    allow_spending: bool = False
    allow_contacting_people: bool = False
    allow_core_self_modification: bool = False
    budget: TaskBudget = field(default_factory=TaskBudget)

    def allows(self, capability: str) -> bool:
        if self.mode == ExecutionMode.NO_TOOLS:
            return False
        if self.mode == ExecutionMode.READ_ONLY and capability.startswith("write_"):
            return False
        if (
            capability.startswith("owner:")
            and capability not in self.owner_approved_capabilities
        ):
            return False
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["capabilities"] = sorted(self.capabilities)
        data["owner_approved_capabilities"] = sorted(
            self.owner_approved_capabilities
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthorizationEnvelope":
        return cls(
            mode=ExecutionMode(data.get("mode", ExecutionMode.FULL_AUTONOMOUS.value)),
            workspace=str(data.get("workspace", "agent_workspace/autonomous")),
            capabilities=set(data.get("capabilities", [])),
            owner_approved_capabilities=set(
                data.get("owner_approved_capabilities", [])
            ),
            protected_paths=tuple(data.get("protected_paths", ())),
            allow_external_models=bool(data.get("allow_external_models", False)),
            allow_spending=bool(data.get("allow_spending", False)),
            allow_contacting_people=bool(data.get("allow_contacting_people", False)),
            allow_core_self_modification=bool(
                data.get("allow_core_self_modification", False)
            ),
            budget=TaskBudget(**data.get("budget", {})),
        )


@dataclass(slots=True)
class AutonomousTask:
    objective: str
    task_id: str = field(default_factory=lambda: uuid4().hex)
    status: TaskStatus = TaskStatus.CREATED
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    action_count: int = 0
    failure_count: int = 0
    checkpoint_sequence: int = 0
    result: str = ""
    blocked_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def transition(self, status: TaskStatus) -> None:
        self.status = status
        if status == TaskStatus.RUNNING and self.started_at is None:
            self.started_at = utc_now()
        if status in {
            TaskStatus.STOPPED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.PANICKED,
        }:
            self.finished_at = utc_now()
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutonomousTask":
        copied = dict(data)
        copied["status"] = TaskStatus(copied.get("status", TaskStatus.CREATED.value))
        return cls(**copied)


@dataclass(slots=True)
class StepResult:
    outcome: StepOutcome
    summary: str = ""
    result: str = ""
    next_instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
