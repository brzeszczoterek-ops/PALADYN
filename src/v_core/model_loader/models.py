from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

from .chat_templates import CHAT_TEMPLATE_PROFILES
from .qualification import ModelQualificationCard


_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GPU_LAYERS = re.compile(r"^(auto|all|[0-9]{1,4})$")
_FLASH_ATTENTION = {"auto", "on", "off"}
_REASONING = {"auto", "on", "off"}
_ANTI_REPETITION = {"off", "balanced", "strong"}
_KV_CACHE_TYPES = {
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q5_0",
    "q5_1",
    "q4_0",
    "q4_1",
    "iq4_nl",
}
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
    "--jinja",
    "--no-jinja",
    "--webui",
    "-ctk",
    "-ctv",
    "--cache-type-k",
    "--cache-type-v",
    "--reasoning",
    "--repeat-last-n",
    "--repeat-penalty",
    "--dry-multiplier",
    "--dry-base",
    "--dry-allowed-length",
    "--dry-penalty-last-n",
    "--metrics",
    "--no-metrics",
    "--slots",
    "--no-slots",
    "-sys",
    "-sysf",
    "--system-prompt",
    "--system-prompt-file",
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
    reasoning: str = "off"
    chat_template: str = "auto"
    anti_repetition: str = "balanced"
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
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
        reasoning = self.reasoning.strip().casefold()
        chat_template = self.chat_template.strip().casefold()
        anti_repetition = self.anti_repetition.strip().casefold()
        cache_type_k = self.cache_type_k.strip().casefold()
        cache_type_v = self.cache_type_v.strip().casefold()
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
        if reasoning not in _REASONING:
            raise ValueError("reasoning must be auto, on, or off")
        if chat_template not in CHAT_TEMPLATE_PROFILES:
            raise ValueError(
                "chat_template must be auto, embedded, or hermes_3_tool_use"
            )
        if anti_repetition not in _ANTI_REPETITION:
            raise ValueError("anti_repetition must be off, balanced, or strong")
        if cache_type_k not in _KV_CACHE_TYPES:
            raise ValueError("cache_type_k is not supported")
        if cache_type_v not in _KV_CACHE_TYPES:
            raise ValueError("cache_type_v is not supported")
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
        object.__setattr__(self, "reasoning", reasoning)
        object.__setattr__(self, "chat_template", chat_template)
        object.__setattr__(self, "anti_repetition", anti_repetition)
        object.__setattr__(self, "cache_type_k", cache_type_k)
        object.__setattr__(self, "cache_type_v", cache_type_v)
        object.__setattr__(self, "extra_args", extras)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        copied = dict(data)
        extras = list(copied.get("extra_args", ()))
        migrated: list[str] = []
        index = 0
        while index < len(extras):
            item = str(extras[index])
            flag, separator, inline_value = item.partition("=")
            target = {
                "-ctk": "cache_type_k",
                "--cache-type-k": "cache_type_k",
                "-ctv": "cache_type_v",
                "--cache-type-v": "cache_type_v",
                "--reasoning": "reasoning",
            }.get(flag)
            if target is None:
                migrated.append(item)
                index += 1
                continue
            if separator:
                value = inline_value
                index += 1
            elif index + 1 < len(extras):
                value = str(extras[index + 1])
                index += 2
            else:
                raise ValueError(f"missing value for {flag}")
            copied.setdefault(target, value)
        copied["extra_args"] = tuple(migrated)
        return cls(**copied)


@dataclass(slots=True)
class LoaderState:
    model_directories: list[str] = field(default_factory=list)
    server_binary: str = ""
    last_model_path: str = ""
    profiles: dict[str, ModelProfile] = field(default_factory=dict)
    routing_enabled: bool = False
    routing_model_paths: list[str] = field(default_factory=list)
    qualifications: dict[str, ModelQualificationCard] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_directories": list(self.model_directories),
            "server_binary": self.server_binary,
            "last_model_path": self.last_model_path,
            "profiles": {
                path: profile.to_dict() for path, profile in self.profiles.items()
            },
            "routing_enabled": self.routing_enabled,
            "routing_model_paths": list(self.routing_model_paths),
            "qualifications": {
                path: card.to_dict() for path, card in self.qualifications.items()
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
        raw_routing_paths = data.get("routing_model_paths", [])
        if not isinstance(raw_routing_paths, list):
            raise ValueError("routing model paths must be an array")
        routing_paths = list(
            dict.fromkeys(str(item) for item in raw_routing_paths if str(item))
        )
        if len(routing_paths) > 3:
            raise ValueError("model router accepts at most three local models")
        raw_qualifications = data.get("qualifications", {})
        if not isinstance(raw_qualifications, dict):
            raise ValueError("model qualifications must be an object")
        if len(raw_qualifications) > 5_000:
            raise ValueError("model loader stores at most 5000 qualifications")
        qualifications = {
            str(path): ModelQualificationCard.from_dict(card)
            for path, card in raw_qualifications.items()
        }
        return cls(
            model_directories=directories,
            server_binary=str(data.get("server_binary", "")),
            last_model_path=str(data.get("last_model_path", "")),
            profiles=profiles,
            routing_enabled=bool(data.get("routing_enabled", False)),
            routing_model_paths=routing_paths,
            qualifications=qualifications,
        )
