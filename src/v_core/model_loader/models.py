from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any


_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GPU_LAYERS = re.compile(r"^(auto|all|[0-9]{1,4})$")
_FLASH_ATTENTION = {"auto", "on", "off"}
_BLOCKED_EXTRA_FLAGS = {
    "-m",
    "-mu",
    "-hf",
    "-hfr",
    "-hff",
    "-a",
    "--model",
    "--model-url",
    "--hf-repo",
    "--hf-file",
    "--host",
    "--port",
    "--alias",
    "--api-key",
    "--models-preset",
    "--tools",
    "--webui",
}


@dataclass(frozen=True, slots=True)
class LocalModel:
    path: Path
    size_bytes: int

    def __post_init__(self) -> None:
        resolved = Path(self.path).expanduser().resolve(strict=True)
        if not resolved.is_file() or resolved.suffix.casefold() != ".gguf":
            raise ValueError("local model must be an existing GGUF file")
        try:
            with resolved.open("rb") as handle:
                magic = handle.read(4)
        except OSError as error:
            raise ValueError("local model cannot be read") from error
        if self.size_bytes < 4 or magic != b"GGUF":
            raise ValueError("local model does not have a GGUF header")
        object.__setattr__(self, "path", resolved)

    @property
    def name(self) -> str:
        return self.path.stem


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_path: str
    alias: str
    context_size: int = 32_768
    gpu_layers: str = "auto"
    threads: int = 0
    batch_size: int = 2_048
    ubatch_size: int = 512
    parallel: int = 1
    flash_attention: str = "auto"
    temperature: float = 0.2
    top_p: float = 0.95
    startup_timeout_seconds: float = 600.0
    port: int = 5_001
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        model_path = str(Path(self.model_path).expanduser().resolve())
        alias = self.alias.strip()
        gpu_layers = str(self.gpu_layers).strip().casefold()
        flash_attention = self.flash_attention.strip().casefold()
        if not _ALIAS.fullmatch(alias):
            raise ValueError("model alias contains unsupported characters")
        if not _GPU_LAYERS.fullmatch(gpu_layers):
            raise ValueError("gpu_layers must be auto, all, or an integer")
        if not 256 <= self.context_size <= 10_000_000:
            raise ValueError("context_size is outside the supported range")
        if not 0 <= self.threads <= 4_096:
            raise ValueError("threads is outside the supported range")
        if not 1 <= self.batch_size <= 1_048_576:
            raise ValueError("batch_size is outside the supported range")
        if not 1 <= self.ubatch_size <= self.batch_size:
            raise ValueError("ubatch_size must be positive and no larger than batch_size")
        if not 1 <= self.parallel <= 1_024:
            raise ValueError("parallel is outside the supported range")
        if flash_attention not in _FLASH_ATTENTION:
            raise ValueError("flash_attention must be auto, on, or off")
        if not 0.0 <= float(self.temperature) <= 5.0:
            raise ValueError("temperature must be between 0 and 5")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
        if not 1.0 <= float(self.startup_timeout_seconds) <= 3_600.0:
            raise ValueError("startup timeout must be between 1 and 3600 seconds")
        if not 1_024 <= self.port <= 65_535:
            raise ValueError("port must be between 1024 and 65535")
        extras = tuple(str(item).strip() for item in self.extra_args)
        if (
            len(extras) > 128
            or sum(len(item.encode("utf-8")) for item in extras) > 64 * 1024
            or any(
                not item or "\x00" in item or len(item.encode("utf-8")) > 4_096
                for item in extras
            )
        ):
            raise ValueError("extra llama.cpp arguments are invalid")
        for item in extras:
            flag = item.split("=", 1)[0]
            if flag in _BLOCKED_EXTRA_FLAGS:
                raise ValueError(f"extra argument cannot override loader boundary: {flag}")
        object.__setattr__(self, "model_path", model_path)
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "gpu_layers", gpu_layers)
        object.__setattr__(self, "flash_attention", flash_attention)
        object.__setattr__(self, "extra_args", extras)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        copied = dict(data)
        copied["extra_args"] = tuple(copied.get("extra_args", ()))
        return cls(**copied)


@dataclass(slots=True)
class LoaderState:
    model_directories: list[str] = field(default_factory=list)
    server_binary: str = ""
    last_model_path: str = ""
    profiles: dict[str, ModelProfile] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_directories": list(self.model_directories),
            "server_binary": self.server_binary,
            "last_model_path": self.last_model_path,
            "profiles": {
                path: profile.to_dict() for path, profile in self.profiles.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoaderState":
        if not isinstance(data, dict):
            raise ValueError("model-loader state must be an object")
        raw_directories = data.get("model_directories", [])
        if not isinstance(raw_directories, list):
            raise ValueError("model directories must be an array")
        directories = [str(item) for item in raw_directories]
        if len(directories) > 64:
            raise ValueError("model loader stores at most 64 directories")
        raw_profiles = data.get("profiles", {})
        if not isinstance(raw_profiles, dict):
            raise ValueError("model profiles must be an object")
        if len(raw_profiles) > 5_000:
            raise ValueError("model loader stores at most 5000 profiles")
        profiles = {
            str(path): ModelProfile.from_dict(profile)
            for path, profile in raw_profiles.items()
        }
        return cls(
            model_directories=directories,
            server_binary=str(data.get("server_binary", "")),
            last_model_path=str(data.get("last_model_path", "")),
            profiles=profiles,
        )
