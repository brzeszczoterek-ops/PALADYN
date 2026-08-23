from __future__ import annotations

import json
import os
from pathlib import Path
import re

from .live import LiveAction, LiveActionGrant


_GRANT_ID = re.compile(r"^[0-9a-f]{32}$")


class LiveGrantStore:
    """Owner-controlled storage outside every autonomous task workspace."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def save(self, grant: LiveActionGrant) -> Path:
        self._validate_id(grant.grant_id)
        path = self.root / f"{grant.grant_id}.json"
        temporary = self.root / f".{grant.grant_id}.tmp"
        payload = {
            "schema_version": 1,
            "grant_id": grant.grant_id,
            "chain_id": grant.chain_id,
            "issued_at": grant.issued_at,
            "expires_at": grant.expires_at,
            "actions": sorted(action.value for action in grant.actions),
            "allowed_targets": sorted(grant.allowed_targets),
            "allowed_selectors": sorted(grant.allowed_selectors),
            "max_value_wei": grant.max_value_wei,
        }
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        return path

    def load(self, grant_id: str, *, now: int | None = None) -> LiveActionGrant | None:
        self._validate_id(grant_id)
        path = self.root / f"{grant_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("grant_id") != grant_id:
            raise ValueError("invalid live grant record")
        grant = LiveActionGrant(
            grant_id=grant_id,
            chain_id=int(payload["chain_id"]),
            issued_at=int(payload["issued_at"]),
            expires_at=int(payload["expires_at"]),
            actions=frozenset(LiveAction(item) for item in payload["actions"]),
            allowed_targets=frozenset(payload.get("allowed_targets", [])),
            allowed_selectors=frozenset(payload.get("allowed_selectors", [])),
            max_value_wei=int(payload.get("max_value_wei", 0)),
        )
        if now is not None and now >= grant.expires_at:
            self.revoke(grant_id)
            return None
        return grant

    def revoke(self, grant_id: str) -> None:
        self._validate_id(grant_id)
        (self.root / f"{grant_id}.json").unlink(missing_ok=True)

    def active(self, *, now: int) -> list[LiveActionGrant]:
        grants: list[LiveActionGrant] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                grant = self.load(path.stem, now=now)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if grant is not None and grant.issued_at <= now:
                grants.append(grant)
        return grants

    @staticmethod
    def _validate_id(grant_id: str) -> None:
        if not _GRANT_ID.fullmatch(grant_id):
            raise ValueError("invalid live grant id")
