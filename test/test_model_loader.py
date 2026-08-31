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
    ModelQualificationCard,
    ModelQualifier,
    ModelRouteCandidate,
    ModelRouter,
    QualificationProbeResult,
    RoutedModelRuntime,
    bootstrap_interactive_model,
    build_server_command,
    choose_model,
    discover_models,
    find_llama_server,
    infer_chat_template,
    start_llama_server,
)
from v_core.llm import LLMResponse
from v_core.persona.primer import V_IDENTITY_PRIMER


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


def qualification_card(
    model: Path,
    profile: ModelProfile,
    **scores: int,
) -> ModelQualificationCard:
    from v_core.model_loader import model_file_fingerprint, model_profile_fingerprint

    capabilities = {
        name: int(scores.get(name, 50))
        for name in (
            "conversation",
            "persona",
            "instruction_following",
            "structured_output",
            "tool_calling",
            "coding",
            "research",
            "grounding",
            "execution_honesty",
        )
    }
    return ModelQualificationCard(
        model_path=str(model.resolve()),
        model_fingerprint=model_file_fingerprint(model),
        profile_fingerprint=model_profile_fingerprint(profile),
        qualified_at="2026-08-31T12:00:00Z",
        harness_version=6,
        capabilities=capabilities,
        probes=(
            QualificationProbeResult(
                name="fixture_probe",
                score=100,
                passed=True,
                latency_ms=1,
                output_digest="a" * 64,
            ),
        ),
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
    with pytest.raises(ValueError, match="chat_template"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            chat_template="download-whatever-the-model-wants",
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
    with pytest.raises(ValueError, match="loader boundary"):
        ModelProfile(
            model_path=str(model),
            alias="test",
            extra_args=("--system-prompt", "replace V"),
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


def test_loader_state_persists_qualified_three_model_routing_pool(
    tmp_path: Path,
) -> None:
    paths = [model_file(tmp_path / f"model-{index}.gguf") for index in range(3)]
    profiles = {str(path.resolve()): profile_for(path) for path in paths}
    cards = {
        key: qualification_card(Path(key), profile)
        for key, profile in profiles.items()
    }
    store = ModelLoaderStore(tmp_path / "runtime")
    state = LoaderState(
        model_directories=[str(tmp_path)],
        profiles=profiles,
        routing_enabled=True,
        routing_model_paths=list(profiles),
        qualifications=cards,
    )

    store.save(state)
    restored = store.load()

    assert restored.to_dict() == state.to_dict()
    with pytest.raises(ValueError, match="at most three"):
        LoaderState.from_dict(
            {
                **state.to_dict(),
                "routing_model_paths": ["a", "b", "c", "d"],
            }
        )


def test_older_qualification_card_remains_readable_but_stale(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "legacy-card.gguf")
    profile = profile_for(model)
    payload = qualification_card(model, profile).to_dict()
    payload["harness_version"] = 5
    payload["capabilities"].pop("grounding")
    payload["capabilities"].pop("execution_honesty")

    restored = ModelQualificationCard.from_dict(payload)

    assert restored.score("grounding") == 0
    assert restored.score("execution_honesty") == 0
    assert restored.is_current(model, profile) is False


@pytest.mark.asyncio
async def test_qualification_harness_scores_protocol_capabilities(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "qualified.gguf")
    profile = profile_for(model)

    class ScriptedLLM:
        async def respond(self, **kwargs):
            prompt = kwargs["messages"][-1]["content"]
            if "PALADYN_READY_731" in prompt:
                return LLMResponse(content="PALADYN_READY_731")
            if "alpha" in prompt:
                return LLMResponse(content='{"alpha":2,"beta":["v"]}')
            if "2+2" in prompt:
                return LLMResponse(content="4")
            if "run(arguments)" in prompt:
                return LLMResponse(
                    content=(
                        "def run(arguments):\n"
                        "    return {\"value\": arguments[\"left\"] + "
                        "arguments[\"right\"]}"
                    )
                )
            if "Authoritative observed evidence" in prompt:
                return LLMResponse(
                    content=(
                        '{"name":"Aster","score":7,"secret_code":null}'
                    )
                )
            if "successful_tool_calls = []" in prompt:
                return LLMResponse(
                    content=(
                        '{"executed":false,"successful_tool_calls":0}'
                    )
                )
            if "local model qualification" in prompt:
                return LLMResponse(
                    content=(
                        '{"name":"web_search","arguments":'
                        '{"query":"local model qualification"}}'
                    )
                )
            if "probe_lookup" in prompt:
                return LLMResponse(
                    content=(
                        '{"name":"probe_lookup","arguments":'
                        '{"query":"paladyn qualification"}}'
                    )
                )
            return LLMResponse(content="Still sharp, Boss.")

    card = await ModelQualifier(ScriptedLLM()).qualify(profile)

    assert card.overall_score == 97
    assert card.score("tool_calling") == 87
    assert card.score("research") == 90
    assert card.score("coding") == 100
    assert card.score("grounding") == 100
    assert card.score("execution_honesty") == 100
    assert card.is_current(model, profile) is True
    changed_profile = ModelProfile.from_dict(
        {**profile.to_dict(), "temperature": 0.9}
    )
    assert card.is_current(model, changed_profile) is False


def test_model_router_selects_verified_specialist_and_keeps_fallbacks(
    tmp_path: Path,
) -> None:
    chat_path = model_file(tmp_path / "chat.gguf")
    code_path = model_file(tmp_path / "code.gguf")
    research_path = model_file(tmp_path / "research.gguf")
    chat_profile = profile_for(chat_path)
    code_profile = profile_for(code_path)
    research_profile = profile_for(research_path)
    candidates = [
        ModelRouteCandidate(
            str(chat_path.resolve()),
            qualification_card(
                chat_path,
                chat_profile,
                conversation=100,
                persona=100,
                coding=20,
                research=20,
                tool_calling=40,
            ),
        ),
        ModelRouteCandidate(
            str(code_path.resolve()),
            qualification_card(
                code_path,
                code_profile,
                conversation=50,
                persona=40,
                coding=100,
                structured_output=100,
                tool_calling=90,
            ),
        ),
        ModelRouteCandidate(
            str(research_path.resolve()),
            qualification_card(
                research_path,
                research_profile,
                conversation=50,
                persona=50,
                research=100,
                tool_calling=100,
                structured_output=90,
            ),
        ),
    ]
    router = ModelRouter()

    chat = router.choose("How are you today?", candidates)
    code = router.choose("Create a Python tool that parses records.", candidates)
    research = router.choose("Search the internet for current llama.cpp docs.", candidates)

    assert chat is not None and chat.selected_model_path == str(chat_path.resolve())
    assert code is not None and code.selected_model_path == str(code_path.resolve())
    assert research is not None
    assert research.selected_model_path == str(research_path.resolve())
    assert len(code.fallback_model_paths) == 2


def test_server_command_is_argument_array_with_enforced_local_boundary(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "model with spaces.gguf")
    profile = profile_for(model)
    primer = tmp_path / "V primer.txt"
    primer.write_text(V_IDENTITY_PRIMER, encoding="utf-8")
    command = build_server_command(
        Path("/usr/bin/true"),
        profile,
        system_prompt_file=primer,
    )

    assert command[0] == "/usr/bin/true"
    assert command[command.index("--model") + 1] == str(model.resolve())
    assert command[-7:] == (
        "--jinja",
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
    assert command[command.index("--system-prompt-file") + 1] == str(
        primer.resolve()
    )
    assert "--chat-template" not in command


def test_hermes_profile_uses_reviewed_offline_tool_template(tmp_path: Path) -> None:
    model = model_file(tmp_path / "Hermes-3-Llama-3.1-8B-lorablated.gguf")
    profile = ModelProfile(
        model_path=str(model),
        alias="hermes-3-test",
        chat_template="auto",
    )
    primer = tmp_path / "primer.txt"
    primer.write_text(V_IDENTITY_PRIMER, encoding="utf-8")

    command = build_server_command(
        Path("/usr/bin/true"),
        profile,
        system_prompt_file=primer,
    )

    assert infer_chat_template(str(model), profile.alias) == "hermes_3_tool_use"
    template = command[command.index("--chat-template") + 1]
    assert "<tools>" in template
    assert "<tool_call>" in template
    assert "function calling AI model" in template
    assert "FunctionCall" in template


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
        primer = tmp_path / "runtime" / "system-prompts" / "v-identity.txt"
        assert primer.read_text(encoding="utf-8").strip() == V_IDENTITY_PRIMER
        assert primer.stat().st_mode & 0o777 == 0o600
        cmdline = Path(f"/proc/{session.process.pid}/cmdline").read_bytes().split(
            b"\0"
        )
        assert b"--system-prompt-file" in cmdline
        assert str(primer.resolve()).encode() in cmdline
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
async def test_loader_reuses_exact_matching_private_llama_server(
    tmp_path: Path,
) -> None:
    model = model_file(tmp_path / "model.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    profile = profile_for(model)
    runtime_root = tmp_path / "runtime"
    first = await start_llama_server(binary, profile, runtime_root)
    try:
        attached = await start_llama_server(binary, profile, runtime_root)
        assert attached.owns_process is True
        assert attached.process.pid == first.process.pid
        assert os.environ["V_CORE_MODEL"] == profile.alias

        await attached.stop()
        assert first.process.poll() is not None
    finally:
        await first.stop()


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
async def test_routed_runtime_unloads_current_model_and_starts_verified_specialist(
    tmp_path: Path,
) -> None:
    chat_model = model_file(tmp_path / "models" / "chat.gguf")
    code_model = model_file(tmp_path / "models" / "code.gguf")
    binary = fake_llama_server(tmp_path / "llama-server")
    shared_port = free_port()
    chat_profile = profile_for(chat_model, port=shared_port)
    code_profile = ModelProfile.from_dict(
        {
            **profile_for(code_model, port=shared_port).to_dict(),
            "alias": "code-model",
        }
    )
    root = tmp_path / "runtime"
    store = ModelLoaderStore(root)
    chat_key = str(chat_model.resolve())
    code_key = str(code_model.resolve())
    store.save(
        LoaderState(
            model_directories=[str(chat_model.parent)],
            server_binary=str(binary),
            last_model_path=chat_key,
            profiles={chat_key: chat_profile, code_key: code_profile},
            routing_enabled=True,
            routing_model_paths=[chat_key, code_key],
            qualifications={
                chat_key: qualification_card(
                    chat_model,
                    chat_profile,
                    conversation=100,
                    persona=100,
                    coding=10,
                    tool_calling=20,
                    structured_output=30,
                ),
                code_key: qualification_card(
                    code_model,
                    code_profile,
                    conversation=30,
                    persona=20,
                    coding=100,
                    tool_calling=100,
                    structured_output=100,
                ),
            },
        )
    )

    class SharedLLM:
        def __init__(self) -> None:
            self.reconfigurations = 0

        async def reconfigure(self) -> None:
            self.reconfigurations += 1

    current = await start_llama_server(binary, chat_profile, root)
    llm = SharedLLM()
    runtime = RoutedModelRuntime(current, root, llm, status=lambda _: None)
    try:
        result = await runtime.ensure_for(
            "Create a Python tool that parses these JSON records."
        )

        assert result.switched is True
        assert result.active_model_path == code_key
        assert current.process.poll() is not None
        assert runtime.session.process.poll() is None
        assert llm.reconfigurations == 1
        assert store.load().last_model_path == code_key
        journal = (root / "routing.jsonl").read_text(encoding="utf-8")
        assert '"task_kind": "coding"' in journal
        assert "Create a Python tool" not in journal
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_routed_runtime_recovers_previous_model_when_specialist_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_model = model_file(tmp_path / "chat.gguf")
    code_model = model_file(tmp_path / "code.gguf")
    chat_profile = profile_for(chat_model)
    code_profile = profile_for(code_model)
    chat_key = str(chat_model.resolve())
    code_key = str(code_model.resolve())
    root = tmp_path / "runtime"
    store = ModelLoaderStore(root)
    store.save(
        LoaderState(
            server_binary="/usr/bin/true",
            last_model_path=chat_key,
            profiles={chat_key: chat_profile, code_key: code_profile},
            routing_enabled=True,
            routing_model_paths=[chat_key, code_key],
            qualifications={
                chat_key: qualification_card(
                    chat_model,
                    chat_profile,
                    conversation=100,
                    persona=100,
                    coding=10,
                ),
                code_key: qualification_card(
                    code_model,
                    code_profile,
                    coding=100,
                    tool_calling=100,
                    structured_output=100,
                ),
            },
        )
    )

    class Process:
        pid = 12345

        def poll(self):
            return None

    class Session:
        def __init__(self, profile):
            self.profile = profile
            self.process = Process()
            self.stopped = False

        async def stop(self):
            self.stopped = True

    starts: list[str] = []

    async def fake_start(binary, profile, runtime_root, *, status):
        starts.append(profile.model_path)
        if profile.model_path == code_key:
            raise LlamaServerStartError("deliberate specialist failure")
        return Session(profile)

    monkeypatch.setattr(
        "v_core.model_loader.routed_runtime.find_llama_server",
        lambda configured: Path("/usr/bin/true"),
    )
    monkeypatch.setattr(
        "v_core.model_loader.routed_runtime.start_llama_server",
        fake_start,
    )

    class SharedLLM:
        async def reconfigure(self):
            pass

    initial = Session(chat_profile)
    runtime = RoutedModelRuntime(initial, root, SharedLLM(), status=lambda _: None)
    result = await runtime.ensure_for("Create a Python tool for these records.")

    assert initial.stopped is True
    assert starts == [code_key, chat_key]
    assert result.active_model_path == chat_key
    assert result.switched is False
    assert any("deliberate specialist failure" in item for item in result.failures)
    await runtime.stop()


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
