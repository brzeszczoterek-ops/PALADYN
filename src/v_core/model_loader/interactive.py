from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import sys
from typing import Callable

from .discovery import discover_models, human_size
from .models import LoaderState, LocalModel, ModelProfile
from .runtime import LlamaServerSession, find_llama_server, start_llama_server
from .storage import ModelLoaderStore


Input = Callable[[str], str]
Output = Callable[[str], None]


class ModelLoaderInteractionError(RuntimeError):
    pass


async def bootstrap_interactive_model(
    runtime_root: Path,
    *,
    mode: str = "prompt",
    input_fn: Input = input,
    output: Output = print,
    stdin_is_tty: bool | None = None,
) -> LlamaServerSession | None:
    selected_mode = mode.strip().casefold()
    if selected_mode not in {"off", "prompt", "required"}:
        raise ValueError("model loader mode must be off, prompt, or required")
    if selected_mode == "off":
        return None
    interactive = sys.stdin.isatty() if stdin_is_tty is None else stdin_is_tty
    if not interactive:
        if selected_mode == "required":
            raise ModelLoaderInteractionError(
                "local model loader requires an interactive terminal"
            )
        return None

    store = ModelLoaderStore(runtime_root)
    state = store.load()
    models = discover_models(state.model_directories)
    while not models:
        output("No local GGUF models were found in the configured directories.")
        directory = _read(
            input_fn,
            "Directory containing GGUF models (Enter = use external server): ",
        ).strip()
        if not directory:
            if selected_mode == "required":
                output("A local model is required for this startup mode.")
                continue
            return None
        resolved = Path(directory).expanduser().resolve()
        if not resolved.is_dir():
            output(f"Directory does not exist: {resolved}")
            continue
        rendered = str(resolved)
        if rendered not in state.model_directories:
            state.model_directories.append(rendered)
        store.save(state)
        models = discover_models(state.model_directories)

    model = choose_model(
        models,
        last_model_path=state.last_model_path,
        input_fn=input_fn,
        output=output,
        allow_external=(selected_mode != "required"),
    )
    if model is None:
        return None

    key = str(model.path)
    saved_profile = state.profiles.get(key)
    profile = (
        saved_profile
        if saved_profile is not None
        and Path(saved_profile.model_path).resolve() == model.path
        else default_profile(model)
    )
    output(render_profile(profile))
    edit = _read(input_fn, "Edit these parameters? [y/N]: ").strip().casefold()
    if edit in {"y", "yes", "t", "tak"}:
        profile = edit_profile(profile, input_fn=input_fn, output=output)

    binary = find_llama_server(state.server_binary)
    while binary is None:
        supplied = _read(
            input_fn,
            "Path to the llama-server executable (Enter = cancel): ",
        ).strip()
        if not supplied:
            if selected_mode == "required":
                output("llama-server is required for this startup mode.")
                continue
            return None
        binary = find_llama_server(supplied)
        if binary is None:
            output("That file is missing or is not executable.")

    state.server_binary = str(binary)
    state.profiles[key] = profile
    store.save(state)
    session = await start_llama_server(binary, profile, runtime_root, status=output)
    try:
        state.last_model_path = key
        store.save(state)
    except BaseException:
        await session.stop()
        raise
    return session


def choose_model(
    models: list[LocalModel],
    *,
    last_model_path: str,
    input_fn: Input,
    output: Output,
    allow_external: bool,
) -> LocalModel | None:
    if not models:
        raise ValueError("cannot choose from an empty model list")
    output("\nLocal GGUF models:")
    default_index = 0
    for index, model in enumerate(models, start=1):
        marker = " (last used)" if str(model.path) == last_model_path else ""
        if marker:
            default_index = index - 1
        output(f"  {index}. {model.path.name} — {human_size(model.size_bytes)}{marker}")
    if allow_external:
        output("  0. Use the server configured in .env")
    while True:
        raw = _read(
            input_fn,
            f"Select model [Enter = {default_index + 1}]: ",
        ).strip()
        if not raw:
            return models[default_index]
        if raw == "0" and allow_external:
            return None
        try:
            index = int(raw) - 1
        except ValueError:
            output("Enter a model number from the list.")
            continue
        if 0 <= index < len(models):
            return models[index]
        output("Selected model number is outside the list.")


def default_profile(model: LocalModel) -> ModelProfile:
    alias = re.sub(r"[^A-Za-z0-9._-]+", "-", model.name).strip("-._")
    alias = (alias or "local-model")[:128]
    return ModelProfile(
        model_path=str(model.path),
        alias=alias,
        context_size=_environment_int("V_CORE_CONTEXT", 32_768),
        temperature=_environment_float("V_CORE_TEMPERATURE", 0.2),
        top_p=_environment_float("V_CORE_TOP_P", 0.95),
        port=_base_url_port(os.getenv("V_CORE_BASE_URL", ""), 5_001),
    )


def edit_profile(
    profile: ModelProfile,
    *,
    input_fn: Input,
    output: Output,
) -> ModelProfile:
    values = profile.to_dict()
    values["alias"] = _prompt_text(input_fn, "API alias", profile.alias)
    values["context_size"] = _prompt_int(
        input_fn, "Context size", profile.context_size, 256, 10_000_000
    )
    values["gpu_layers"] = _prompt_choice_or_integer(
        input_fn, "GPU layers", profile.gpu_layers, {"auto", "all"}, 0, 9_999
    )
    values["threads"] = _prompt_int(
        input_fn, "CPU threads (0 = llama.cpp default)", profile.threads, 0, 4_096
    )
    values["batch_size"] = _prompt_int(
        input_fn, "Batch size", profile.batch_size, 1, 1_048_576
    )
    values["ubatch_size"] = _prompt_int(
        input_fn, "Micro-batch size", profile.ubatch_size, 1, values["batch_size"]
    )
    values["parallel"] = _prompt_int(
        input_fn, "Parallel slots", profile.parallel, 1, 1_024
    )
    values["flash_attention"] = _prompt_choice(
        input_fn,
        "Flash attention",
        profile.flash_attention,
        {"auto", "on", "off"},
    )
    values["temperature"] = _prompt_float(
        input_fn, "Temperature", profile.temperature, 0.0, 5.0
    )
    values["top_p"] = _prompt_float(
        input_fn, "Top-p", profile.top_p, 0.000001, 1.0
    )
    values["port"] = _prompt_int(input_fn, "Local API port", profile.port, 1_024, 65_535)
    values["startup_timeout_seconds"] = _prompt_float(
        input_fn,
        "Startup timeout in seconds",
        profile.startup_timeout_seconds,
        1.0,
        3_600.0,
    )
    rendered_extras = shlex.join(profile.extra_args)
    extra_text = _prompt_text(
        input_fn,
        "Additional llama.cpp arguments",
        rendered_extras,
    )
    try:
        values["extra_args"] = tuple(shlex.split(extra_text)) if extra_text else ()
        return ModelProfile.from_dict(values)
    except (ValueError, TypeError) as error:
        output(f"Invalid profile: {error}")
        output("Keeping the previous profile.")
        return profile


def render_profile(profile: ModelProfile) -> str:
    return (
        "\nSelected profile:\n"
        f"  alias={profile.alias}\n"
        f"  context={profile.context_size}, gpu_layers={profile.gpu_layers}, "
        f"threads={profile.threads or 'auto'}\n"
        f"  batch={profile.batch_size}, ubatch={profile.ubatch_size}, "
        f"parallel={profile.parallel}, flash_attn={profile.flash_attention}\n"
        f"  temperature={profile.temperature:g}, top_p={profile.top_p:g}, "
        f"port={profile.port}\n"
        f"  extra_args={shlex.join(profile.extra_args) or '(none)'}"
    )


def _read(input_fn: Input, prompt: str) -> str:
    try:
        return input_fn(prompt)
    except EOFError:
        return ""


def _prompt_text(input_fn: Input, label: str, default: str) -> str:
    value = _read(input_fn, f"{label} [{default or 'none'}]: ").strip()
    return value or default


def _prompt_int(
    input_fn: Input,
    label: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    while True:
        raw = _read(input_fn, f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            continue
        if minimum <= value <= maximum:
            return value


def _prompt_float(
    input_fn: Input,
    label: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    while True:
        raw = _read(input_fn, f"{label} [{default:g}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            continue
        if minimum <= value <= maximum:
            return value


def _prompt_choice(
    input_fn: Input,
    label: str,
    default: str,
    choices: set[str],
) -> str:
    while True:
        value = _read(input_fn, f"{label} [{default}]: ").strip().casefold()
        if not value:
            return default
        if value in choices:
            return value


def _prompt_choice_or_integer(
    input_fn: Input,
    label: str,
    default: str,
    choices: set[str],
    minimum: int,
    maximum: int,
) -> str:
    while True:
        value = _read(input_fn, f"{label} [{default}]: ").strip().casefold()
        if not value:
            return default
        if value in choices:
            return value
        try:
            parsed = int(value)
        except ValueError:
            continue
        if minimum <= parsed <= maximum:
            return str(parsed)


def _environment_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _environment_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _base_url_port(value: str, default: int) -> int:
    try:
        from urllib.parse import urlparse

        return urlparse(value).port or default
    except ValueError:
        return default
