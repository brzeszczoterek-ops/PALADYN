from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket

import pytest

from v_core.model_loader import (
    LoaderState,
    LocalModel,
    LlamaServerStartError,
    ModelLoaderStore,
    ModelProfile,
    bootstrap_interactive_model,
    build_server_command,
    choose_model,
    discover_models,
    find_llama_server,
    start_llama_server,
)


def model_file(path: Path, size: int = 16) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"x" * max(0, size - 4))
    return path


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def fake_llama_server(path: Path, *, exit_immediately: bool = False) -> Path:
    if exit_immediately:
        source = "#!/usr/bin/python3\nprint('deliberate loader failure', flush=True)\nraise SystemExit(7)\n"
    else:
        source = r'''#!/usr/bin/python3
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys

def value(flag, default=""):
    positions = [index for index, item in enumerate(sys.argv) if item == flag]
    return sys.argv[positions[-1] + 1] if positions else default

alias = value("--alias", "local-model")
port = int(value("--port", "8080"))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            payload = {"status": "ok"}
            status = 200
        elif self.path == "/v1/models":
            payload = {"object": "list", "data": [{"id": alias}]}
            status = 200
        else:
            payload = {"error": "not found"}
            status = 404
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''
    path.write_text(source, encoding="utf-8")
    path.chmod(0o700)
    return path


def profile_for(model: Path, *, port: int | None = None) -> ModelProfile:
    return ModelProfile(
        model_path=str(model),
        alias="test-model",
        context_size=4_096,
        gpu_layers="all",
        threads=4,
        batch_size=512,
        ubatch_size=128,
        parallel=2,
        flash_attention="on",
        reasoning="off",
        anti_repetition="balanced",
        cache_type_k="q8_0",
        cache_type_v="q4_0",
        temperature=0.3,
        top_p=0.9,
        startup_timeout_seconds=5,
        port=port or free_port(),
        extra_args=("--mlock",),
    )


def test_discovery_finds_nested_models_and_filters_auxiliary_gguf(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models"
    first = model_file(root / "alpha.GGUF", 10)
    shard = model_file(root / "nested" / "beta-00001-of-00002.gguf", 20)
    model_file(root / "nested" / "beta-00002-of-00002.gguf", 20)
    model_file(root / "nested" / "mmproj-beta.gguf", 5)
    (root / "fake.gguf").write_bytes(b"this is not GGUF")
    (root / "not-a-model.bin").write_bytes(b"x")

    found = discover_models([root])

    assert [item.path for item in found] == [first.resolve(), shard.resolve()]
    assert [item.size_bytes for item in found] == [10, 20]


def test_profile_rejects_boundary_overrides(tmp_path: Path) -> None:
    model = model_file(tmp_path / "model.gguf")
    with pytest.raises(ValueError, match="loader boundary"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            extra_args=("--host=0.0.0.0",),
        )
    with pytest.raises(ValueError, match="ubatch_size"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            batch_size=16,
            ubatch_size=32,
        )
    with pytest.raises(ValueError, match="cache_type_k"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            cache_type_k="q2_bogus",
        )
    with pytest.raises(ValueError, match="reasoning"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            reasoning="sometimes",
        )
    with pytest.raises(ValueError, match="anti_repetition"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            anti_repetition="maximum",
        )
    with pytest.raises(ValueError, match="loader boundary"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            extra_args=("--cache-type-v", "q4_0"),
        )


def test_old_extra_kv_cache_arguments_migrate_to_profile_fields(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "model.gguf")
    profile = ModelProfile.from_dict(
        {
            "model_path": str(model),
            "alias": "test",
            "extra_args": [
                "--poll",
                "100",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v=q4_0",
                "--reasoning",
                "off",
            ],
        }
    )

    assert profile.cache_type_k == "q8_0"
    assert profile.cache_type_v == "q4_0"
    assert profile.reasoning == "off"
    assert profile.extra_args == ("--poll", "100")


def test_loader_state_round_trips_atomically_with_private_permissions(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "model.gguf")
    profile = profile_for(model)
    store = ModelLoaderStore(tmp_path / "runtime")
    state = LoaderState(
        model_directories=[str(tmp_path)],
        server_binary="/usr/bin/true",
        last_model_path=str(model.resolve()),
        profiles={str(model.resolve()): profile},
    )

    store.save(state)
    restored = store.load()

    assert restored.to_dict() == state.to_dict()
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert store.root.stat().st_mode & 0o777 == 0o700


def test_server_command_is_argument_array_with_enforced_local_boundary(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "model with spaces.gguf")
    profile = profile_for(model)
    command = build_server_command(Path("/usr/bin/true"), profile)

    assert command[0] == "/usr/bin/true"
    assert command[command.index("--model") + 1] == str(model.resolve())
    assert command[-6:] == (
        "--metrics",
        "--slots",
        "--offline",
        "--no-webui",
        "--host",
        "127.0.0.1",
    )
    assert "--mlock" in command
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--cache-type-v") + 1] == "q4_0"
    assert command[command.index("--reasoning") + 1] == "off"
    assert command[command.index("--repeat-last-n") + 1] == "256"
    assert command[command.index("--repeat-penalty") + 1] == "1.08"
    assert command[command.index("--dry-multiplier") + 1] == "0.8"


def test_find_llama_server_honors_explicit_executable(tmp_path: Path) -> None:
    binary = fake_llama_server(tmp_path / "llama-server")
    assert find_llama_server(str(binary), environ={}) == binary.resolve()


def test_choose_model_uses_last_model_as_default_and_can_select_external(
    tmp_path: Path,
) -> None:
    first_path = model_file(tmp_path / "a.gguf")
    second_path = model_file(tmp_path / "b.gguf")
    models = [LocalModel(first_path, 16), LocalModel(second_path, 16)]
    output: list[str] = []

    selected = choose_model(
        models,
        last_model_path=str(second_path.resolve()),
        input_fn=lambda _: "",
        output=output.append,
        allow_external=True,
    )
    external = choose_model(
        models,
        last_model_path="",
        input_fn=lambda _: "0",
        output=output.append,
        allow_external=True,
    )

    assert selected is not None and selected.path == second_path.resolve()
    assert external is None
    assert any("last used" in line for line in output)


@pytest.mark.asyncio
async def test_fake_llama_server_starts_verifies_alias_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_file(tmp_path / "model.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    profile = profile_for(model)

    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    session = await start_llama_server(binary, profile, tmp_path / "runtime")
    try:
        assert session.process.poll() is None
        assert os.environ["V_CORE_BASE_URL"] == session.base_url
        assert os.environ["V_CORE_MODEL"] == "test-model"
        assert os.environ["V_CORE_CONTEXT"] == "4096"
        assert session.log_path.exists()
    finally:
        await session.stop()

    assert session.process.poll() is not None


@pytest.mark.asyncio
async def test_loader_reports_early_server_exit_with_log_tail(tmp_path: Path) -> None:
    model = model_file(tmp_path / "model.gguf")
    binary = fake_llama_server(tmp_path / "llama-server", exit_immediately=True)
    profile = profile_for(model)

    with pytest.raises(LlamaServerStartError, match="deliberate loader failure"):
        await start_llama_server(binary, profile, tmp_path / "runtime")


@pytest.mark.asyncio
async def test_loader_refuses_to_replace_service_on_selected_port(
    tmp_path: Path,
) -> None:
    async def handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        await reader.read(8_192)
        body = b'{"status":"ok"}'
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    model = model_file(tmp_path / "model.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    try:
        with pytest.raises(LlamaServerStartError, match="already has an HTTP service"):
            await start_llama_server(
                binary,
                profile_for(model, port=port),
                tmp_path / "runtime",
            )
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_interactive_bootstrap_runs_complete_saved_profile_flow(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "models" / "model.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    profile = profile_for(model)
    runtime_root = tmp_path / "runtime"
    store = ModelLoaderStore(runtime_root)
    store.save(
        LoaderState(
            model_directories=[str(model.parent)],
            server_binary=str(binary),
            profiles={str(model.resolve()): profile},
        )
    )
    answers = iter(("", "n"))

    session = await bootstrap_interactive_model(
        runtime_root,
        mode="prompt",
        input_fn=lambda _: next(answers),
        output=lambda _: None,
        stdin_is_tty=True,
    )
    assert session is not None
    try:
        assert session.profile == profile
        assert store.load().last_model_path == str(model.resolve())
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_noninteractive_loader_skips_prompt_or_fails_when_required(
    tmp_path: Path,
) -> None:
    assert (
        await bootstrap_interactive_model(
            tmp_path / "prompt",
            mode="prompt",
            stdin_is_tty=False,
        )
        is None
    )
    with pytest.raises(RuntimeError, match="interactive terminal"):
        await bootstrap_interactive_model(
            tmp_path / "required",
            mode="required",
            stdin_is_tty=False,
        )
