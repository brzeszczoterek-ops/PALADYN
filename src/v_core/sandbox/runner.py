from __future__ import annotations

from dataclasses import dataclass, replace
from v_core.autonomy import AuthorizationDenied, AuthorizationGuard

from .bubblewrap import BubblewrapBackend
from .models import SandboxResult, SandboxSpec


@dataclass(slots=True)
class SandboxExecutor:
    """Authorization-aware bridge between an autonomous task and a backend."""

    authorization: AuthorizationGuard
    backend: BubblewrapBackend

    async def execute(self, spec: SandboxSpec) -> SandboxResult:
        self.authorization.require("run_sandboxed_code")
        workspace = self.authorization.resolve_task_path(spec.workspace, write=True)

        if spec.read_only_inputs:
            raise AuthorizationDenied(
                "autonomous tasks must stage inputs inside their task workspace"
            )

        authorized_spec = replace(
            spec,
            workspace=workspace,
            read_only_inputs=(),
        )
        return await self.backend.run(authorized_spec)
