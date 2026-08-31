from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from .models import ModelProfile
from .qualification import ModelQualificationCard
from .router import ModelRouteCandidate, ModelRouteDecision, ModelRouter
from .runtime import (
    LlamaServerSession,
    LlamaServerStartError,
    find_llama_server,
    start_llama_server,
)
from .storage import ModelLoaderStore


@dataclass(frozen=True, slots=True)
class ModelSwitchResult:
    decision: ModelRouteDecision | None
    previous_model_path: str
    active_model_path: str
    switched: bool
    failures: tuple[str, ...] = ()


class RoutedModelRuntime:
    """Own one local server and replace it only between top-level user turns."""

    def __init__(
        self,
        session: LlamaServerSession,
        runtime_root: Path,
        llm: Any,
        *,
        status: Callable[[str], None] = print,
    ) -> None:
        self.session = session
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.store = ModelLoaderStore(self.runtime_root)
        self.llm = llm
        self.status = status
        self.router = ModelRouter()
        self._journal_path = self.runtime_root / "routing.jsonl"

    @property
    def active_model_path(self) -> str:
        return str(Path(self.session.profile.model_path).expanduser().resolve())

    async def ensure_for(
        self,
        prompt: str,
        *,
        task_kind: str | None = None,
        excluded_model_paths: Iterable[str] = (),
        trigger: str = "task_route",
    ) -> ModelSwitchResult:
        state = self.store.load()
        previous = self.active_model_path
        if not state.routing_enabled or not state.routing_model_paths:
            return ModelSwitchResult(None, previous, previous, False)

        candidates = self._current_candidates(state)
        decision = self.router.choose(
            prompt,
            candidates,
            current_model_path=previous,
            task_kind=task_kind,
            excluded_model_paths=excluded_model_paths,
        )
        if decision is None or decision.selected_model_path == previous:
            result = ModelSwitchResult(decision, previous, previous, False)
            self._record(prompt, result, trigger=trigger)
            return result

        binary = find_llama_server(state.server_binary)
        if binary is None:
            failure = "configured llama-server is unavailable"
            result = ModelSwitchResult(
                decision,
                previous,
                previous,
                False,
                (failure,),
            )
            self._record(prompt, result, trigger=trigger)
            return result

        profiles = state.profiles
        ordered = list(
            dict.fromkeys(
                (
                    decision.selected_model_path,
                    *decision.fallback_model_paths,
                    previous,
                )
            )
        )
        previous_profile = self.session.profile
        await self.session.stop()
        failures: list[str] = []
        for path in ordered:
            profile = profiles.get(path)
            if path == previous and profile is None:
                profile = previous_profile
            if profile is None:
                failures.append(f"{Path(path).name}: saved profile is missing")
                continue
            session: LlamaServerSession | None = None
            try:
                session = await start_llama_server(
                    binary,
                    profile,
                    self.runtime_root,
                    status=self.status,
                )
                await self.llm.reconfigure()
            except Exception as error:
                if session is not None:
                    await session.stop()
                failures.append(f"{Path(path).name}: {type(error).__name__}: {error}")
                continue
            self.session = session
            state.last_model_path = path
            self.store.save(state)
            result = ModelSwitchResult(
                decision,
                previous,
                path,
                path != previous,
                tuple(failures),
            )
            if path != previous:
                self.status(
                    f"V routed this {decision.task_kind} task to "
                    f"{Path(path).name} (verified score {decision.score}/100)."
                )
            self._record(prompt, result, trigger=trigger)
            return result

        raise LlamaServerStartError(
            "model routing stopped the active server and every qualified fallback "
            "failed to start: " + "; ".join(failures)
        )

    async def ensure_for_phase(
        self,
        prompt: str,
        task_kind: str,
    ) -> ModelSwitchResult:
        """Route at a runtime-owned phase boundary inside one mixed task."""

        return await self.ensure_for(
            prompt,
            task_kind=task_kind,
            trigger="phase_route",
        )

    async def retry_after_rejection(
        self,
        prompt: str,
        task_kind: str,
    ) -> ModelSwitchResult:
        """Retry a failed response with the next qualified local specialist.

        Model output cannot name its own replacement. The runtime excludes the
        model that just failed and selects the next model from persisted,
        current qualification cards for the same task class.
        """

        rejected = self.active_model_path
        return await self.ensure_for(
            prompt,
            task_kind=task_kind,
            excluded_model_paths=(rejected,),
            trigger="response_rejection",
        )

    async def stop(self) -> None:
        await self.session.stop()

    def _current_candidates(self, state: Any) -> list[ModelRouteCandidate]:
        candidates: list[ModelRouteCandidate] = []
        for path in state.routing_model_paths[:3]:
            profile: ModelProfile | None = state.profiles.get(path)
            card: ModelQualificationCard | None = state.qualifications.get(path)
            if profile is None or card is None:
                continue
            model_path = Path(path).expanduser()
            if not card.is_current(model_path, profile):
                continue
            candidates.append(ModelRouteCandidate(path, card))
        return candidates

    def _record(
        self,
        prompt: str,
        result: ModelSwitchResult,
        *,
        trigger: str,
    ) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "trigger": trigger,
            "task_kind": result.decision.task_kind if result.decision else "unrouted",
            "previous_model_path": result.previous_model_path,
            "active_model_path": result.active_model_path,
            "switched": result.switched,
            "score": result.decision.score if result.decision else 0,
            "failures": list(result.failures),
        }
        descriptor = os.open(
            self._journal_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(
                descriptor,
                (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
                    "utf-8"
                ),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
