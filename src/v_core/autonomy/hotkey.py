from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import struct


EV_KEY = 0x01
KEY_RELEASED = 0
KEY_PRESSED = 1
KEY_REPEATED = 2

LINUX_KEY_CODES = {
    "0": 11,
    "Q": 16,
    "P": 25,
    "NUM0": 82,
}
_INPUT_EVENT = struct.Struct("llHHi")


def parse_chord(value: str) -> frozenset[str]:
    keys = frozenset(part.strip().upper() for part in value.split("+") if part.strip())
    if len(keys) < 2:
        raise ValueError("emergency chord must contain at least two different keys")
    unsupported = keys - LINUX_KEY_CODES.keys()
    if unsupported:
        raise ValueError(f"unsupported emergency keys: {', '.join(sorted(unsupported))}")
    return keys


@dataclass(slots=True)
class ChordDetector:
    required: frozenset[str] = field(
        default_factory=lambda: frozenset({"Q", "P", "0"})
    )
    pressed: set[str] = field(default_factory=set)
    triggered: bool = False

    def feed(self, key: str, is_pressed: bool) -> bool:
        normalized = key.upper()
        if is_pressed:
            self.pressed.add(normalized)
        else:
            self.pressed.discard(normalized)

        complete = self.required <= self.pressed
        if complete and not self.triggered:
            self.triggered = True
            return True
        if not complete:
            self.triggered = False
        return False


@dataclass(slots=True)
class LinuxHotkeyWatcher:
    device: Path
    chord: frozenset[str] = field(
        default_factory=lambda: frozenset({"Q", "P", "0"})
    )

    def watch(self, on_trigger: Callable[[], None]) -> None:
        detector = ChordDetector(self.chord)
        reverse_codes = {
            code: name
            for name, code in LINUX_KEY_CODES.items()
            if name in self.chord
        }
        for event_type, code, value in self.events():
            if event_type != EV_KEY or code not in reverse_codes:
                continue
            if value not in {KEY_RELEASED, KEY_PRESSED, KEY_REPEATED}:
                continue
            if detector.feed(reverse_codes[code], value != KEY_RELEASED):
                on_trigger()

    def events(self) -> Iterator[tuple[int, int, int]]:
        with Path(self.device).open("rb", buffering=0) as handle:
            while data := handle.read(_INPUT_EVENT.size):
                if len(data) != _INPUT_EVENT.size:
                    continue
                _, _, event_type, code, value = _INPUT_EVENT.unpack(data)
                yield event_type, code, value


def input_devices() -> list[tuple[Path, str]]:
    devices: list[tuple[Path, str]] = []
    for event in sorted(Path("/sys/class/input").glob("event*")):
        name_path = event / "device" / "name"
        try:
            name = name_path.read_text(encoding="utf-8").strip()
        except OSError:
            name = "unknown input device"
        devices.append((Path("/dev/input") / event.name, name))
    return devices
