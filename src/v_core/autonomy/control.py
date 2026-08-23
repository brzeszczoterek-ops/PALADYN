from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from .journal import validate_task_id
from .models import ControlSignal


class StopRequested(Exception):
    pass


class PanicRequested(Exception):
    pass


@dataclass(slots=True)
class GlobalControlChannel:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self.root / "_global"

    @property
    def panic_path(self) -> Path:
        return self.directory / ControlSignal.PANIC.value

    def request_panic(self) -> Path:
        self.panic_path.write_text(ControlSignal.PANIC.value, encoding="utf-8")
        return self.panic_path

    def clear_panic(self) -> None:
        if self.panic_path.exists():
            self.panic_path.unlink()

    def panic_pending(self) -> bool:
        return self.panic_path.exists()


@dataclass(slots=True)
class ControlChannel:
    root: Path
    task_id: str
    poll_interval: float = 0.25

    def __post_init__(self) -> None:
        validate_task_id(self.task_id)
        self.root = Path(self.root)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self.root / self.task_id

    def request(self, signal: ControlSignal) -> Path:
        path = self.directory / signal.value
        path.write_text(signal.value, encoding="utf-8")
        return path

    def clear(self, signal: ControlSignal) -> None:
        path = self.directory / signal.value
        if path.exists():
            path.unlink()

    def clear_all(self) -> None:
        for signal in ControlSignal:
            self.clear(signal)

    def pending(self) -> set[ControlSignal]:
        pending = {
            signal
            for signal in ControlSignal
            if (self.directory / signal.value).exists()
        }
        if GlobalControlChannel(self.root).panic_pending():
            pending.add(ControlSignal.PANIC)
        return pending

    async def checkpoint(self) -> bool:
        pending = self.pending()

        if ControlSignal.PANIC in pending:
            raise PanicRequested("panic requested by owner")
        if ControlSignal.STOP in pending:
            raise StopRequested("stop requested by owner")

        if ControlSignal.PAUSE not in pending:
            return False

        while True:
            await asyncio.sleep(self.poll_interval)
            pending = self.pending()

            if ControlSignal.PANIC in pending:
                raise PanicRequested("panic requested while paused")
            if ControlSignal.STOP in pending:
                raise StopRequested("stop requested while paused")
            if ControlSignal.RESUME in pending:
                self.clear(ControlSignal.PAUSE)
                self.clear(ControlSignal.RESUME)
                return True
