from __future__ import annotations

from pathlib import Path
import re

from .models import LocalModel


_LATER_SHARD = re.compile(r"-(?!00001-)\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


def discover_models(
    directories: list[str | Path],
    *,
    maximum: int = 5_000,
) -> list[LocalModel]:
    if not 1 <= maximum <= 100_000:
        raise ValueError("model discovery maximum is outside the supported range")
    found: dict[Path, LocalModel] = {}
    for directory_value in directories:
        directory = Path(directory_value).expanduser().resolve()
        if not directory.is_dir():
            continue
        try:
            candidates = directory.rglob("*")
            for path in candidates:
                try:
                    if not path.is_file() or path.suffix.casefold() != ".gguf":
                        continue
                    if "mmproj" in path.name.casefold() or _LATER_SHARD.search(path.name):
                        continue
                    resolved = path.resolve(strict=True)
                    size = resolved.stat().st_size
                    if size <= 0:
                        continue
                    if resolved not in found and len(found) >= maximum:
                        raise RuntimeError(
                            f"model discovery exceeded the limit of {maximum} GGUF files"
                        )
                    found.setdefault(resolved, LocalModel(resolved, size))
                except (OSError, ValueError):
                    continue
        except OSError:
            continue
    return sorted(found.values(), key=lambda item: item.path.name.casefold())


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1_024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1_024
    return f"{value:.1f} TiB"
