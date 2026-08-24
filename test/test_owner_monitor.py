from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

from v_core.owner_monitor import (
    build_monitor_command,
    compact_tegrastats,
    counter_rate,
    launch_owner_monitor,
    monitor_session_identity,
    parse_prometheus,
    parse_response_timing,
    TelemetryJournal,
)


def _session(tmp_path: Path):
    return SimpleNamespace(
        profile=SimpleNamespace(
            port=5001,
            alias="mythos",
            context_size=32768,
            reasoning="off",
            cache_type_k="q8_0",
            cache_type_v="q4_0",
        ),
        process=SimpleNamespace(pid=4321),
        log_path=tmp_path / "runtime" / "logs" / "llama-server-20260824-012340.log",
    )


def test_prometheus_parser_ignores_comments_and_aggregates_labels() -> None:
    metrics = parse_prometheus(
        """
# HELP llamacpp:tokens_predicted_total Generated tokens
llamacpp:tokens_predicted_total 42
llamacpp:requests_processing{slot="0"} 1
llamacpp:requests_processing{slot="1"} 2
not-a-number nope
"""
    )

    assert metrics["llamacpp:tokens_predicted_total"] == 42
    assert metrics["llamacpp:requests_processing"] == 3
    assert "not-a-number" not in metrics


def test_timing_parser_uses_latest_completed_response() -> None:
    timing = parse_response_timing(
        """
slot print_timing: prompt eval time = 100.00 ms / 100 tokens (1.00 ms per token, 1000.00 tokens per second)
slot print_timing: eval time = 500.00 ms / 20 tokens (25.00 ms per token, 40.00 tokens per second)
slot print_timing: total time = 600.00 ms / 120 tokens
slot print_timing: prompt eval time = 200.00 ms / 50 tokens (4.00 ms per token, 250.00 tokens per second)
slot print_timing: eval time = 1000.00 ms / 10 tokens (100.00 ms per token, 10.00 tokens per second)
slot print_timing: total time = 1200.00 ms / 60 tokens
"""
    )

    assert timing.prompt_tokens == 50
    assert timing.prompt_tokens_per_second == 250.0
    assert timing.generated_tokens == 10
    assert timing.generated_tokens_per_second == 10.0
    assert timing.total_milliseconds == 1200.0


def test_counter_rate_uses_model_processing_time() -> None:
    previous = {
        "llamacpp:tokens_predicted_total": 20,
        "llamacpp:tokens_predicted_seconds": 4,
    }
    current = {
        "llamacpp:tokens_predicted_total": 32,
        "llamacpp:tokens_predicted_seconds": 6,
    }

    assert counter_rate(
        current,
        previous,
        token_suffix="tokens_predicted_total",
        seconds_suffix="tokens_predicted_seconds",
    ) == 6.0


def test_tegrastats_is_compacted_for_narrow_terminal() -> None:
    hardware = compact_tegrastats(
        "RAM 4790/15598MB SWAP 122/30720MB "
        "CPU [20%@1420,40%@1420] GR3D_FREQ 25% "
        "cpu@40.875C gpu@39.281C VDD_IN 9498mW"
    )

    assert hardware == (
        "RAM 4790/15598 MB  ·  SWAP 122/30720 MB",
        "CPU 30%  ·  GPU 25%  ·  temp CPU 40.875°C / GPU 39.281°C",
        "Power 9.50 W",
    )


def test_monitor_command_contains_only_explicit_session_values(tmp_path: Path) -> None:
    session = _session(tmp_path)
    command = build_monitor_command(
        session,
        terminal=Path("/usr/bin/gnome-terminal"),
        python=Path("/venv/bin/python"),
    )

    assert command[0] == "/usr/bin/gnome-terminal"
    assert command[command.index("--pid") + 1] == "4321"
    assert command[command.index("--model") + 1] == "mythos"
    assert command[command.index("--cache-v") + 1] == "q4_0"
    assert command[command.index("--log") + 1] == str(session.log_path)
    assert command[command.index("--session") + 1] == "20260824-012340-pid4321"
    assert command[command.index("--telemetry") + 1].endswith(
        "monitor_sessions/20260824-012340-pid4321.jsonl"
    )


def test_monitor_session_identity_is_unique_and_bound_to_current_log(
    tmp_path: Path,
) -> None:
    session_id, telemetry = monitor_session_identity(_session(tmp_path))

    assert session_id == "20260824-012340-pid4321"
    assert telemetry == (
        tmp_path
        / "runtime"
        / "monitor_sessions"
        / "20260824-012340-pid4321.jsonl"
    )


def test_telemetry_journal_is_private_append_only_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "monitor_sessions" / "session.jsonl"
    journal = TelemetryJournal(path, "session-1")
    journal.append("session_start", {"model": "mythos"})
    journal.append("sample", {"generated_tokens_total": 42.0})
    journal.close()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["session_start", "sample"]
    assert all(event["session_id"] == "session-1" for event in events)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_owner_monitor_is_disabled_by_default(tmp_path: Path) -> None:
    called = False

    def fake_popen(*args, **kwargs):
        nonlocal called
        called = True

    assert (
        launch_owner_monitor(_session(tmp_path), environ={}, popen=fake_popen)
        is False
    )
    assert called is False


def test_owner_monitor_launches_when_owner_flag_is_enabled(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], dict]] = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace()

    launched = launch_owner_monitor(
        _session(tmp_path),
        environ={
            "PALADYN_OWNER_MONITOR": "1",
            "PALADYN_OWNER_TERMINAL": "/usr/bin/gnome-terminal",
        },
        popen=fake_popen,
    )

    assert launched is True
    assert len(calls) == 1
    assert calls[0][0][0] == "/usr/bin/gnome-terminal"
    assert calls[0][1]["start_new_session"] is True
