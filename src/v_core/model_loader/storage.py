from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .models import LoaderState


class ModelLoaderStorageError(RuntimeError):
    pass


class ModelLoaderStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.path = self.root / "loader.json"

    def load(self) -> LoaderState:
        if not self.path.exists():
            return LoaderState()
        try:
            if self.path.stat().st_size > 2_000_000:
                raise ModelLoaderStorageError("model-loader state exceeds 2 MB")
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ModelLoaderStorageError("model-loader document must be an object")
            if payload.get("schema_version") != 1:
                raise ModelLoaderStorageError("unsupported model-loader schema")
            return LoaderState.from_dict(payload.get("state", {}))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            if isinstance(error, ModelLoaderStorageError):
                raise
            raise ModelLoaderStorageError(
                f"could not read model-loader state: {error}"
            ) from error

    def save(self, state: LoaderState) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".loader-",
            suffix=".tmp",
            dir=self.root,
        )
        temporary = Path(temporary_name)
        try:
            os.chmod(temporary, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {"schema_version": 1, "state": state.to_dict()},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self.path.chmod(0o600)
        except Exception as error:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise ModelLoaderStorageError(
                f"could not save model-loader state: {error}"
            ) from error
