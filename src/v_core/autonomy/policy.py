from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import AuthorizationEnvelope


class AuthorizationDenied(PermissionError):
    pass


@dataclass(slots=True)
class AuthorizationGuard:
    project_root: Path
    envelope: AuthorizationEnvelope

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()

    def require(self, capability: str) -> None:
        if not self.envelope.allows(capability):
            raise AuthorizationDenied(
                f"capability is outside the authorization envelope: {capability}"
            )

        if capability == "use_external_model" and not self.envelope.allow_external_models:
            raise AuthorizationDenied("external models are disabled")
        if capability == "spend_money" and not self.envelope.allow_spending:
            raise AuthorizationDenied("spending is disabled")
        if capability == "contact_people" and not self.envelope.allow_contacting_people:
            raise AuthorizationDenied("contacting people is disabled")

    def resolve_task_path(
        self,
        requested: str | Path,
        *,
        write: bool,
    ) -> Path:
        workspace = self._resolve(self.envelope.workspace)
        requested_path = Path(requested)
        path = (
            requested_path.resolve()
            if requested_path.is_absolute()
            else (workspace / requested_path).resolve()
        )

        if not path.is_relative_to(workspace):
            raise AuthorizationDenied("path escapes the autonomous task workspace")

        if write:
            self.require("write_task_workspace")
            self._reject_protected(path)
        else:
            self.require("read_workspace")

        return path

    def _reject_protected(self, path: Path) -> None:
        if self.envelope.allow_core_self_modification:
            return

        for protected in self.envelope.protected_paths:
            protected_path = self._resolve(protected)
            if path == protected_path or path.is_relative_to(protected_path):
                raise AuthorizationDenied(f"protected path: {protected}")

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.project_root / candidate).resolve()
